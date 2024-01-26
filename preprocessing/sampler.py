import geopandas as gpd
import numpy as np
import shapely

import config


class RandomTrainingTrainingDataSampler:
    def __init__(self) -> None:
        self.index = gpd.read_file(config.env("BAG3D_SHEET_INDEX"))
        self.cities = gpd.read_file(config.env("CITIES"))
        # TODO: Harmonize this variable name.
        self.randm = np.random.default_rng(int(config.var("SEED")))

    # TODO: Find out why the override decorator cannot be imported from the typing
    #       module.
    def sample(self, size: int = 10):
        sample = []
        while len(sample) < size:
            city_pt = self.cities.sample(random_state=self.randm)["geometry"]

            # TODO: Refactor this line.
            tile_pt = gpd.GeoDataFrame(
                {"id": [0], "geometry": [self._gen_random_point(city_pt)]},
                crs="EPSG:28992",
            )
            # NOTE: The point can be positioned on the interface of two or more
            #       adjacent tiles.
            tile_ids = self.index.overlay(tile_pt, keep_geom_type=False)["id_1"]
            for id_ in tile_ids:
                # TODO: Replace this linear search with a more efficient method.
                if id_ in sample:
                    continue
                sample.append(id_)
        return sample

    def _gen_random_point(
        self, pt: gpd.GeoSeries, radius: float = 10000
    ) -> shapely.Point:
        # noinspection PyTypeChecker
        off = self.randm.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        off /= np.linalg.norm(off)
        off *= radius
        return shapely.Point(pt.x + off[0], pt.y + off[1])
