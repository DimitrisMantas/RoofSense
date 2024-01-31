from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Optional

import laspy
import laspy.lasappender
import numpy as np
import pandas as pd
import rasterio
import shapely
import sklearn.neighbors

from parsers.utils import raster
from utils.type import BoundingBoxLike


class PointCloud:
    def __init__(self, path: str | PathLike) -> None:
        with laspy.open(path) as f:
            self.las = f.read()

        self.index = None

    # TODO: Add type hints to this method.
    def __getitem__(self, key):
        if isinstance(key, int):
            # Fetch a single point record.
            # NOTE: This indexing notation ensures that this method consistently
            #       returns a PackedPointRecord instance.
            k = [key]
        else:
            k = key
        return self.points[k]

    def __len__(self) -> int:
        return len(self.points)

    @property
    def header(self) -> laspy.LasHeader:
        return self.las.header

    @property
    def points(self) -> laspy.PackedPointRecord:
        return self.las.points

    @property
    def bbox(self) -> BoundingBoxLike:
        return (
            self.header.x_min,
            self.header.y_min,
            self.header.x_max,
            self.header.y_max,
        )

    def crop(self, bbox: BoundingBoxLike) -> PointCloud:
        # NOTE: This operation must be performed using the original point records to
        #       avoid floating-point comparisons.
        #       The output is not affected because the transformation from records to
        #       coordinates is linear.
        xmin = (bbox[0] - self.header.x_offset) / self.header.x_scale
        ymin = (bbox[1] - self.header.y_offset) / self.header.y_scale
        xmax = (bbox[2] - self.header.x_offset) / self.header.x_scale
        ymax = (bbox[3] - self.header.y_offset) / self.header.y_scale
        self.las.points = self.las.points[
            np.logical_and(
                np.logical_and(xmin <= self["X"], self["X"] <= xmax),
                np.logical_and(ymin <= self["Y"], self["Y"] <= ymax),
            )
        ]
        return self

    def remove_duplicates(self) -> PointCloud:
        pts = np.vstack((self["X"], self["Y"], self["Z"])).transpose()
        # NOTE: The points are sorted in descending elevation order so that the
        #       highest ones are preserved.
        pts = pts[np.argsort(-pts[:, 2])]
        # NOTE: The unique element filter provided by NumPy is inefficient.
        #       https://github.com/numpy/numpy/issues/11136
        self.las.points = self.las.points[
            pd.DataFrame(pts).drop_duplicates(subset=[0, 1]).index.tolist()
        ]
        return self

    # TODO: Parallelize this method.
    # TODO: Optimize the search radius, power, and fill search radius hyperparameters.
    def rasterize(
        self,
        scalars: str | Sequence[str],
        resol: float,
        meta: Optional[rasterio.profiles.Profile] = None,
    ) -> dict[str, raster.Raster]:
        if isinstance(scalars, str):
            scalars = [scalars]
        # FIXME: Refactor ``xy()`` into a module function because doesn't make sense to
        #        create an empty raster.Raster just for the sake of using its accessors.
        r = raster.Raster(resol, self.bbox)

        self._init_index()
        neighbors, distances = self.index.query_radius(
            r.xy(), r=resol, return_distance=True
        )

        rasters = {
            scalar: raster.Raster(resol, bbox=self.bbox, meta=meta)
            for scalar in scalars
        }
        # FIXME: Review the rest of this method.
        for cell_id, cell_nb in enumerate(neighbors):
            if len(cell_nb) == 0:
                continue
            attribs = {scalar: [] for scalar in scalars}
            for scalar in scalars:
                attribs[scalar].append(self[cell_nb][scalar])
            row, col = np.divmod(cell_id, r.width)
            if np.any(distances[cell_id] == 0):
                # The cell center belongs to the point cloud.
                nb_id = cell_nb[np.argsort(distances[cell_id])[0]]
                for scalar in scalars:
                    rasters[scalar][row, col] = self[[nb_id]][scalar].scaled_array()
            else:
                weights = distances[cell_id] ** -2
                for scalar in scalars:
                    rasters[scalar][row, col] = np.sum(
                        attribs[scalar] * weights
                    ) / np.sum(weights)

        for raster.Raster in rasters.values():
            raster.Raster.fill()

        return rasters

    # noinspection PyTypeChecker
    def save(self, path: Sequence[str | PathLike]) -> None:
        with laspy.open(path, "w", header=self.header) as f:
            f.write_points(self.las.points)

    def _init_index(self):
        if self.index is None:
            # noinspection PyUnresolvedReferences
            self.index = sklearn.neighbors.KDTree(self.las.xyz)


# noinspection PyTypeChecker
def merge(
    ipaths: Sequence[str | PathLike],
    opath: str | PathLike,
    crop: Optional[BoundingBoxLike] = None,
    remove_duplicates: bool = False,
) -> None:
    out_header = _init_out_file(ipaths[0], opath)

    with laspy.open(opath, "a") as f:
        # Initialize the output header.
        # NOTE: The initial bounding box of the merged output should be defined
        #       around its region so that it can grow correctly.
        f.header.mins = out_header.mins
        f.header.maxs = out_header.maxs

        for path in ipaths:
            tile = PointCloud(path)
            # Translate the local coordinate system of the point cloud that of the
            # output file.
            tile.las.change_scaling(f.header.scales, f.header.offsets)
            if crop is not None:
                tile.crop(crop)
            f.append_points(tile.points)
    if remove_duplicates:
        PointCloud(opath).remove_duplicates().save(opath)


def _init_out_file(ipath: str | PathLike, opath: str | PathLike) -> laspy.LasHeader:
    with laspy.open(ipath) as f:
        ohead = f.header
        laspy.open(opath, "w", header=ohead).close()
    return ohead


# TODO: Consider adopting shapely.Polygon to avoid spaghetti code.
def _remove_overlap(tile: PointCloud, visited_bboxes: list[BoundingBoxLike]) -> None:
    tile_bbox: shapely.Polygon
    tile_bbox = shapely.box(*tile.bbox)
    for bbox in visited_bboxes:
        tile_bbox -= tile_bbox.intersection(bbox)
    # NOTE: The bounding box of the tile must be marked as visited and stored in its
    #       original state so that its neighbors can be processed correctly.
    visited_bboxes.append(shapely.box(*tile.bbox))
    tile.crop(tile_bbox.bounds)


if __name__ == "__main__":
    import config
    import utils

    config.config()
    obj_id = "9-284-556"
    surfs = utils.geom.buffer(utils.geom.read_surfaces(obj_id))
    paths = [
        "C:/Documents/RoofSense/data/temp/37EN1_15.LAZ",
        "C:/Documents/RoofSense/data/temp/37EN1_20.LAZ",
        "C:/Documents/RoofSense/data/temp/37EN2_11.LAZ",
        "C:/Documents/RoofSense/data/temp/37EN2_16.LAZ",
    ]
    # for i, perm in enumerate(itertools.permutations(paths)):
    #     print(f"{i}/{24}", perm)
    merge(
        paths, "test.merged.gtxm.laz", crop=surfs.total_bounds, remove_duplicates=True
    )
