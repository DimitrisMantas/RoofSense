from abc import ABC, abstractmethod

import geopandas as gpd
import numpy as np
import shapely
from typing_extensions import override

import config


class DataSampler(ABC):
    def __init__(self) -> None:
        self._rng = np.random.default_rng(int(config.var("SEED")))

    @abstractmethod
    def sample(self, size: int):
        ...


class BAG3DSampler(DataSampler):
    def __init__(self) -> None:
        super().__init__()
        self._index = gpd.read_file(config.env("BAG3D_SHEET_INDEX"))
        self._seeds = gpd.read_file(config.env("CITIES"))

    @override
    def sample(self, size: int = 10):
        sample = []
        while len(sample) < size:
            seed_pt = self._seeds.sample(random_state=self._rng)[
                config.var("DEFAULT_GM_FIELD_NAME")
            ]
            tile_pt = gpd.GeoDataFrame(
                {
                    config.var("DEFAULT_ID_FIELD_NAME"): [0],
                    config.var("DEFAULT_GM_FIELD_NAME"): [
                        self._gen_random_point(seed_pt)
                    ],
                },
                crs=config.var("CRS"),
            )
            # NOTE: The point can be positioned on the interface of two or more
            #       adjacent tiles.
            tile_ids = self._index.overlay(tile_pt, keep_geom_type=False)["id_1"]
            for tile_id in tile_ids:
                if tile_id in sample:
                    continue
                # Discard tiles whose surface are is larger than 100 ha.
                # NOTE: This limit is padded by 10% to account for any discrepancies in
                #       the underlying sheet index.
                # NOTE: This ensures that at most four AHN4 tiles are downloaded and
                #       parsed per tile,
                #       and thus a minimum level of service is maintained during the
                #       preprocessing stage.
                # NOTE: The selected tile IDs begin with 9 or 10.
                if (self._index.loc[self._index[
                                        config.var("DEFAULT_ID_FIELD_NAME")] == tile_id, config.var(
                    "DEFAULT_GM_FIELD_NAME"),].area.iat[0] > 1.1e6):
                    continue
                sample.append(tile_id)
        return sample

    def _gen_random_point(self, seed: shapely.Point, radius: float = 15e3
    ) -> shapely.Point:
        # noinspection PyTypeChecker
        off = self._rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        nrm = np.linalg.norm(off)
        if nrm > 1:
            off /= np.linalg.norm(off)
        off *= radius
        return shapely.Point(seed.x + off[0], seed.y + off[1])
