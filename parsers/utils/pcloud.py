from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Optional, Self

import laspy
import numpy as np
import polars as pl
import rasterio
import sklearn.neighbors

from parsers.utils import raster
from utils.type import BoundingBoxLike


class PointCloud:
    def __init__(self, path: str | PathLike) -> None:
        with laspy.open(path) as f:
            self.las = f.read()

        self._index = None

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

    def crop(self, bbox: BoundingBoxLike) -> Self:
        xmin = (bbox[0] - self.header.x_offset) / self.header.x_scale
        ymin = (bbox[1] - self.header.y_offset) / self.header.y_scale
        xmax = (bbox[2] - self.header.x_offset) / self.header.x_scale
        ymax = (bbox[3] - self.header.y_offset) / self.header.y_scale
        self.las.points = self.las.points[
            np.logical_and(
                np.logical_and(xmin <= self.points["X"], self.points["X"] <= xmax),
                np.logical_and(ymin <= self.points["Y"], self.points["Y"] <= ymax),
            )
        ]
        return self

    def remove_duplicates(self) -> Self:
        pts = np.vstack(
            [
                np.arange(len(self.points["X"])),
                self.points["X"],
                self.points["Y"],
                self.points["Z"],
            ]
        ).transpose()
        # NOTE: The unique element filter provided by NumPy is inefficient.
        #       See https://github.com/numpy/numpy/issues/11136 for more information.
        #       In addition,
        #       this approach does not disturb the internal record order
        #       and is multithreaded.
        pts = pl.DataFrame(pts, schema=["I", "X", "Y", "Z"])
        # NOTE: The point records are sorted by their elevation to ensure that only
        #       contextually irrelevant points will be discarded.
        pts = pts.sort("Z")
        self.las.points = self.las.points[
            pts.unique(subset=["X", "Y"], keep="last")["I"].to_numpy()
        ]
        return self

    # FIXME: Parallelize this method.
    # TODO: Optimize the interpolation and postprocessing hyperparameters.
    def rasterize(
        self,
        scalars: str | Sequence[str],
        # Rasterisation Options
        resol: float,
        bbox: Optional[BoundingBoxLike] = None,
        meta: Optional[rasterio.profiles.Profile] = None,
        # TODO: Expose the interpolation and postprocessing options.
        **kwargs,
    ) -> dict[str, raster.Raster]:
        self._init_index()

        _bbox = bbox if bbox is not None else self.bbox
        if isinstance(scalars, str):
            # NOTE: This indexing notations ensures consistent output.
            scalars = [scalars]

        rasters = {
            scalar: raster.Raster(resol, bbox=_bbox, meta=meta) for scalar in scalars
        }

        # Interpolation

        # Create a reference raster of the same size as the output images to enable
        # efficient access to each cell.
        ref_ras = raster.Raster(resol, _bbox)
        neighbors, distances = self._index.query_radius(
            ref_ras.xy(), r=resol, return_distance=True
        )
        for cell_id, cell_neighbors in enumerate(neighbors):
            if len(cell_neighbors) == 0:
                continue
            attribs = {scalar: [] for scalar in scalars}
            for scalar in scalars:
                attribs[scalar].append(self.points[cell_neighbors][scalar])

            row, col = np.divmod(cell_id, ref_ras.width)
            if np.any(distances[cell_id] == 0):
                # The cell center is a member of the point cloud.
                pt_id = cell_neighbors[np.argsort(distances[cell_id])[0]]
                for scalar in scalars:
                    rasters[scalar][row, col] = self.points[[pt_id]][
                        scalar
                    ].scaled_array()
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
        # Postprocessing
        for r in rasters.values():
            r.fill(**kwargs)
        return rasters

    def save(self, path: Sequence[str | PathLike]) -> None:
        # noinspection PyTypeChecker
        with laspy.open(path, "w", header=self.header) as f:
            f.write_points(self.las.points)

    def _init_index(self):
        if self._index is not None:
            return
        pts = np.vstack([self.points["x"], self.points["y"]]).transpose()
        # noinspection PyUnresolvedReferences
        self._index = sklearn.neighbors.KDTree(pts)

    def _rasterize(
        self,
        scalar: str,
        resol: float,
        bbox: Optional[BoundingBoxLike] = None,
        meta: Optional[rasterio.profiles.Profile] = None,
    ):
        raise NotImplementedError


def merge(
    ipaths: Sequence[str | PathLike],
    opath: str | PathLike,
    crop: Optional[BoundingBoxLike] = None,
    rem_dpls: bool = False,
) -> None:
    opc = PointCloud(ipaths[0])
    if crop is not None:
        opc.crop(crop)
    for path in ipaths[1:]:
        tmp = PointCloud(path)
        if crop is not None:
            tmp.crop(crop)
        tmp.las.change_scaling(opc.header.scales, opc.header.offsets)
        opc.las.points = laspy.ScaleAwarePointRecord(
            np.concatenate([tmp.points.array, opc.points.array]),
            opc.header.point_format,
            scales=opc.header.scales,
            offsets=opc.header.offsets,
        )
    if rem_dpls:
        opc.remove_duplicates()
    opc.save(opath)
