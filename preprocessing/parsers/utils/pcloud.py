from __future__ import annotations

import math
from collections.abc import Sequence
from io import BytesIO
from os import PathLike
from typing import Optional, Self

import laspy
import numpy as np
import polars as pl
import rasterio
import startinpy
from laspy import LasData, LasHeader, PackedPointRecord
from tqdm import tqdm

from preprocessing.parsers.utils import raster
from utils.type import BoundingBoxLike


class PointCloud:
    def __init__(self, filepath: str | bytes | BytesIO) -> None:
        with laspy.open(filepath) as src:
            self._data = src.read()
        # NOTE: The DT is used as a cheap spatial index, so it is initialized and
        # populated only when required to avoid increased loading times and memory
        # consumption.
        self._index: Optional[startinpy.DT] = None

    @property
    def bbox(self) -> BoundingBoxLike:
        """The two-dimensional, coordinate interleaved, axis-aligned bounding box of
        the point cloud: [xmin, ymin, xmax, ymax]."""
        return *self.header.mins[:2], *self.header.maxs[:2]

    @property
    def data(self) -> LasData:
        """The header, point, and variable length records of the underlying file. See
        :class:`LasData` for more information."""
        return self._data

    @property
    def header(self) -> LasHeader:
        """The header of the underlying file. See :class:`LasHeader` for more
        information."""
        return self.data.header

    @property
    def points(self) -> PackedPointRecord:
        """The point record of the underlying file. See :class:`PackedPointRecord`
        for more information."""
        return self.data.points

    def crop(self, bbox: BoundingBoxLike) -> Self:
        # Map the bounding box to point record space.
        # NOTE: This approach avoids potentially unreliable floating-point numerical
        # operations and results in reduced memory consumption because only point
        # coordinates must be computed on demand.
        xmin = int((bbox[0] - self.header.x_offset) / self.header.x_scale)
        ymin = int((bbox[1] - self.header.y_offset) / self.header.y_scale)
        xmax = int((bbox[2] - self.header.x_offset) / self.header.x_scale)
        ymax = int((bbox[3] - self.header.y_offset) / self.header.y_scale)

        self.data.points = self.points[
            np.logical_and(
                np.logical_and(xmin <= self.points["X"], self.points["X"] <= xmax),
                np.logical_and(ymin <= self.points["Y"], self.points["Y"] <= ymax),
            )
        ]

        return self

    def normals(self) -> Self:
        raise NotImplementedError

    def rasterize(
        self,
        scalar: str,
        resol: float,
        bbox: Optional[
            BoundingBoxLike
        ] = None,
        meta: Optional[
            rasterio.profiles.Profile
        ] = None,
    ) -> raster.Raster:
        self._init_index()
        self._updt_index(scalar)

        _bbox = bbox if bbox is not None else self.bbox

        # Main Loop
        out = raster.Raster(
            resol,
            _bbox,
            meta
        )
        # dat=np.empty((len(out),))
        # for i,cell in tqdm(
        #         enumerate(out.xy()),
        #         desc="Rasterization Progress",
        #         total=len(out)
        # ):
        #     dat[i]=self._index.interpolate(
        #         {"method": "Laplace"},
        #         [cell]
        #     )
        # out._data=dat.reshape((out.height,out.width))
        with tqdm(
                desc="Rasterization Progress",
                total=len(out)
        ) as pbar:
            for row in range(out.height):
                y = _bbox[3] - resol*(row+0.5)
                for col in range(out.width):
                    x = _bbox[0] +  resol*(col+0.5)
                    out[row, col] = self._index.interpolate(
                        {"method": "Laplace"},
                        [[x, y]]
                    )
                    pbar.update()

        return out

    def remove_duplicates(self) -> Self:
        pts = np.vstack(
            [
                self.points["X"],
                self.points["Y"],
                self.points["Z"],
            ]
        ).transpose()

        # NOTE: The unique element filter provided by NumPy is inefficient.
        #       See https://github.com/numpy/numpy/issues/11136 for more information.
        #       In addition,
        #       this approach does not disturb the spatial coherence of the point cloud
        #       and is multithreaded.
        pts = pl.DataFrame(pts, schema=["X", "Y", "Z"])
        pts = pts.with_row_index()

        # Sort the point records by their elevation.
        # NOTE: This ensures that only contextually irrelevant points are discarded.
        pts = pts.sort("Z")

        self.data.points = self.data.points[
            pts.unique(subset=["X", "Y"], keep="last")["index"].sort().to_numpy()
        ]

        return self

    def save(self, filepath: Sequence[str | bytes | PathLike]) -> None:
        # noinspection PyTypeChecker
        with laspy.open(filepath, mode="w", header=self.header) as dst:
            dst.write_points(self.data.points)

    def _init_index(self) -> None:
        if self._index is not None:
            return

        self._index = startinpy.DT()
        # NOTE: This is important to have 1-1 mapping between the DT and PC vertices.
        self._index.snap_tolerance = math.ulp(0)
        for pt in tqdm(self.data.xyz, desc="Index Initialization"):
            self._index.insert_one_pt(pt)

        assert len(self._index.points[1:]) == len(self.points)

    def _updt_index(self, scalar: str) -> None:
        # FIXME: Update the index only iff required.
        attrs = self.points[scalar].scaled_array()
        for i, attr in tqdm(enumerate(attrs), desc="Index Update", total=len(attrs)):
            self._index.update_vertex_z_value(i, attr)


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
        tmp.data.change_scaling(opc.header.scales, opc.header.offsets)
        opc.data.points = laspy.ScaleAwarePointRecord(
            np.concatenate([opc.points.array, tmp.points.array]),
            opc.header.point_format,
            scales=opc.header.scales,
            offsets=opc.header.offsets,
        )
    if rem_dpls:
        opc.remove_duplicates()
    opc.save(opath)


if __name__ == "__main__":
    pc = PointCloud("../../../lidar.laz")
    pc.rasterize("z",resol=0.25).save("lidar.elev.1.tiff")
