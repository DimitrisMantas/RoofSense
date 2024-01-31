from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Optional

import laspy
import laspy.lasappender
import numpy as np
import shapely
import sklearn.neighbors

from parsers.utils import raster
from utils.type import BoundingBoxLike


class PointCloud:
    def __init__(self, path: str | PathLike) -> None:
        self.index = None
        with laspy.open(path) as f:
            self.las = f.read()

    def __len__(self) -> int:
        return len(self.points)

    # TODO: Add type hints to this method.
    def __getitem__(self, key):
        if isinstance(key, int):
            # Get the first point.
            # NOTE: This indexing notation results in the point being returned in the
            #       same format regardless of the key type.
            k = [key]
        else:
            k = key
        return self.points[k]

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
        xmin = (bbox[0] - self.header.x_offset) / self.header.x_scale
        ymin = (bbox[1] - self.header.y_offset) / self.header.y_scale
        xmax = (bbox[2] - self.header.x_offset) / self.header.x_scale
        ymax = (bbox[3] - self.header.y_offset) / self.header.y_scale
        self.las.points = self.las.points[
            np.logical_and(
                np.logical_and(xmin <= self.las.X, self.las.X <= xmax),
                np.logical_and(ymin <= self.las.Y, self.las.Y <= ymax),
            )
        ]
        return self

    # FIXME: Parallelize this function.
    # TODO: Add type hints to this function.
    # TODO: Optimize the search radius, power, and filler hyperparameters.
    def rasterize(
        self, scalars: str | Sequence[str], size: float
    ) -> raster.Raster | dict[str, raster.Raster]:
        if isinstance(scalars, str):
            scalars = [scalars]
        # FIXME: Refactor ``xy()`` into a module function because doesn't make sense to
        #        create an empty raster.Raster just for the sake of using its accessors.
        r = raster.Raster(size, self.bbox)

        # TODO: The point cloud should ensure that its spatial index has been initialized
        #       before this call.
        if self.index is None:
            # noinspection PyUnresolvedReferences
            self.index = sklearn.neighbors.KDTree(
                np.vstack((self.las.x, self.las.y)).transpose()
            )
        neighbors, distances = self.index.query_radius(
            r.xy(), r=size, return_distance=True
        )

        rasters = {scalar: raster.Raster(size, self.bbox) for scalar in scalars}
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


# noinspection PyTypeChecker
def merge(
    in_paths: Sequence[str | PathLike],
    out_path: str | PathLike,
    crop: Optional[BoundingBoxLike] = None,
    overlapping_tiles: bool = False,
) -> None:
    out_header = _init_out_file(in_paths, out_path)

    with laspy.open(out_path, "a") as f:
        # Initialize the output header.
        # NOTE: The initial bounding box of the merged output should be defined
        #       around its region so that it can grow correctly.
        f.header.mins = out_header.mins
        f.header.maxs = out_header.maxs

        visited_bboxes = []
        for path in in_paths:
            tile = PointCloud(path)
            # Translate the local coordinate system of the point cloud that of the
            # output file.
            tile.las.change_scaling(f.header.scales, f.header.offsets)
            if crop is not None:
                tile.crop(crop)
            if overlapping_tiles:
                _remove_overlap(tile, visited_bboxes)
            f.append_points(tile.points)


def _init_out_file(
    in_paths: Sequence[str | PathLike], out_path: str | PathLike
) -> laspy.LasHeader:
    with laspy.open(in_paths[0]) as f:
        out_header = f.header
        laspy.open(out_path, "w", header=out_header).close()
    return out_header


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

    config.config()
    pc = PointCloud("bk.laz").rasterize("z", size=0.25)["z"].save("bk.z.tiff")
