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

    def __getitem__(self, key: int | Sequence[int] | str | Sequence[str]):
        # FIXME: This is NOT working!
        try:
            if np.issubdtype(key, np.integer):
                # Fetch a single point record.
                # NOTE: This indexing notation ensures consistent output.
                k = [key]
        except TypeError:
            pass
        finally:
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
                np.logical_and(xmin <= self["X"], self["X"] <= xmax),
                np.logical_and(ymin <= self["Y"], self["Y"] <= ymax),
            )
        ]
        return self

    def remove_duplicates(self) -> PointCloud:
        pts = np.vstack([self["X"], self["Y"], self["Z"]]).transpose()
        # NOTE: The point records are sorted in descending elevation order to ensure
        #       that the output cloud will contain the points most likely
        #       to correspond to roof surfaces.
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
        **kwargs,
    ) -> dict[str, raster.Raster]:
        self._init_index()
        if isinstance(scalars, str):
            # NOTE: This indexing notations ensures consistent output.
            scalars = [scalars]

        # Create a temporary raster of the same size as the output images to enable
        # efficient access to each cell.
        tmp = raster.Raster(resol, self.bbox)
        neighbors, distances = self.index.query_radius(
            tmp.xy(), r=resol, return_distance=True
        )

        rasters = {
            scalar: raster.Raster(resol, bbox=self.bbox, meta=meta)
            for scalar in scalars
        }
        for cell_id, cell_neighbors in enumerate(neighbors):
            if len(cell_neighbors) == 0:
                continue

            attribs = {scalar: [] for scalar in scalars}
            for scalar in scalars:
                attribs[scalar].append(self[cell_neighbors][scalar])

            row, col = np.divmod(cell_id, tmp.width)

            if np.any(distances[cell_id] == 0):
                # The cell center is a member of the point cloud.
                pt_id = cell_neighbors[np.argsort(distances[cell_id])[0]]
                for scalar in scalars:
                    rasters[scalar][row, col] = self[[pt_id]][scalar].scaled_array()
            else:
                weights = distances[cell_id] ** -2
                for scalar in scalars:
                    # NOTE: The output value cannot be computed using the corresponding
                    #       method provided by NumPy
                    #       because the attribute array contains ScaledArrayView
                    #       instances.
                    rasters[scalar][row, col] = np.sum(
                        attribs[scalar] * weights
                    ) / np.sum(weights)

        for r in rasters.values():
            r.fill(**kwargs)

        return rasters

    # noinspection PyTypeChecker
    def save(self, path: Sequence[str | PathLike]) -> None:
        with laspy.open(path, "w", header=self.header) as f:
            f.write_points(self.las.points)

    # noinspection PyUnresolvedReferences
    def _init_index(self):
        if self.index is not None:
            return
        pts = np.vstack([self["x"], self["y"]]).transpose()
        self.index = sklearn.neighbors.KDTree(pts)


def merge(
    ipaths: Sequence[str | PathLike],
    opath: str | PathLike,
    crop: Optional[BoundingBoxLike] = None,
    remove_duplicates: bool = False,
) -> None:
    out_header = _init_out_file(ipaths[0], opath)

    with laspy.open(opath, "a") as f:
        # Initialize the output header.
        # NOTE: The initial bounding box of the output point cloud must be defined
        #       around its region to ensure that it will grow correctly.
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
    #       original state
    #       to ensure that its neighbors will be processed correctly.
    visited_bboxes.append(shapely.box(*tile.bbox))
    tile.crop(tile_bbox.bounds)
