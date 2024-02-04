from abc import ABC, abstractmethod

import geopandas as gpd
import numpy as np
import shapely
from typing_extensions import override

import config


class DataSamplerABC(ABC):
    def __init__(self) -> None:
        self.rand = np.random.default_rng(int(config.var("SEED")))

    @abstractmethod
    def sample(self, size: int):
        ...


class DataSampler(DataSamplerABC):
    def __init__(self) -> None:
        super().__init__()

        self.index = gpd.read_file(config.env("BAG3D_SHEET_INDEX"))
        self.cities = gpd.read_file(config.env("CITIES"))

    @override
    def sample(self, size: int = 10):
        sample = []
        while len(sample) < size:
            city_pt = self.cities.sample(random_state=self.rand)["geometry"]

            # TODO: Refactor this line.
            tile_pt = gpd.GeoDataFrame(
                {"id": [0], "geometry": [self._gen_random_point(city_pt)]},
                crs="EPSG:28992",
            )
            # NOTE: The point can be positioned on the interface of two or more
            #       adjacent tiles.
            tile_ids = self.index.overlay(tile_pt, keep_geom_type=False)["id_1"]
            for id_ in tile_ids:
                # TODO: Check whether there is a significant performance improvement
                #       between using linear search compared to a hash-based approach.
                if id_ in sample:
                    continue
                sample.append(id_)
        return sample

    def _gen_random_point(
        self, pt: gpd.GeoSeries, radius: float = 10000
    ) -> shapely.Point:
        # noinspection PyTypeChecker
        off = self.rand.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        nrm = np.linalg.norm(off)
        if nrm > 1:
            off /= np.linalg.norm(off)
        off *= radius
        return shapely.Point(pt.x + off[0], pt.y + off[1])
