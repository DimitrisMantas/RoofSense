from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from io import BytesIO
from os import PathLike
from typing import Any, Literal, Optional, Self

import laspy
import numpy as np
import polars as pl
import rasterio
import startinpy
from tqdm import tqdm

from preprocessing.parsers.utils import raster
from utils.type import BoundingBoxLike


class PointCloud:
    """Convenience class for manipulating point clouds in the `LASer
    <https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format
    -exchange-activities>`_ and `LASzip <https://rapidlasso.de/laszip/>`_ file
    formats."""

    def __init__(
        self,
        filepath: str | bytes | BytesIO,
        init_index: bool = False,
        duplicate_predicate: Literal["First", "Highest", "Last", "Lowest"] = "Highest",
        duplicate_tolerance: float = 0.02,
    ) -> None:
        """Initialize the point cloud.

        :param filepath: The path to the file containing the point cloud.

        :param init_index: A boolean flag indicating whether to index the point cloud
        upon its initialization. Regardless of the value of this flag, a spatial
        index is created automatically, if one does not already exist, to support all
        operations whose performance would significantly benefit from optimized
        spatial access methods.

        :param duplicate_predicate: The duplicate point handling strategy used by the spatial index of the point cloud. Given a group of duplicate points in  :math:`\mathcal{R}^{2}`, as specified by

        :param duplicate_tolerance:
        """
        # Initialize the cloud.
        with laspy.open(filepath) as src:
            self._data = src.read()

        # Initialize the index.
        self._index = (
            Index(
                self,
                duplicate_predicate=duplicate_predicate,
                duplicate_tolerance=duplicate_tolerance,
            )
            if init_index
            else None
        )

        # NOTE: The index configuration must be persisted] for it to be able to be
        #       initialized at a later time.
        self._duplicate_predicate = duplicate_predicate
        self._duplicate_tolerance = duplicate_tolerance

    def __len__(self) -> int:
        """Get the number of points in the point cloud."""
        return len(self.data)

    @property
    def bbox(self) -> BoundingBoxLike:
        """Get the two-dimensional, coordinate interleaved, axis-aligned bounding
        box, :math:`[x_{min}, y_{min}, x_{max}, y_{max}]`, specifying the spatial
        extent of the point cloud in the length unit of its coordinate reference
        system."""
        return *self.header.mins[:2], *self.header.maxs[:2]

    @property
    def data(self) -> laspy.LasData:
        """Get the header, point, variable, and extended variable length records of
        the file containing the point cloud.

        .. note::
            See :class:`laspy.LasData` for more information.
        """
        return self._data

    @property
    def index(self) -> Optional[Index]:
        """Get the spatial index of the point cloud.

        :return: The index or ``None`` if it has not been initialized.
        """
        return self._index

    @property
    def header(self) -> laspy.LasHeader:
        """Get the header of the file containing the point cloud.

        .. note::
            See :class:`laspy.LasHeader` for more information.
        """
        return self.data.header

    @property
    def points(self) -> laspy.PackedPointRecord:
        """Get the point record of the file containing the point cloud.

        .. note::
        See :class:`laspy.PackedPointRecord` for more information.
        """
        return self.data.points

    def crop(self, bbox: BoundingBoxLike) -> Self:
        """Crop the point cloud.

        .. warning::
            This operation permanently alters the point cloud!


        :param bbox: the two-dimensional, coordinate interleaved, axis-aligned
        bounding box, :math:`[x_{min}, y_{min}, x_{max}, y_{max}]`, specifying the
        new spatial extent of the point cloud in the length unit of its coordinate
        reference system.


        :return: The decimated point cloud.
        """
        # Map the bounding box to the point record space.
        # NOTE: This approach avoids potentially unreliable floating-point operations
        #       and allows for increased performance and reduced memory consumption
        #       because point coordinates are not stored in the point cloud and must be
        #       computed on demand.
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

        # Invalidate the index.
        if self.index is not None:
            self._index = Index(
                self,
                duplicate_predicate=self.index.duplicates_handling,
                duplicate_tolerance=self.index.snap_tolerance,
            )

        return self

    def normals(self) -> Self:
        # Initialize the index.
        # TODO: Factor this block out to a private method.
        if self.index is None:
            self._index = Index(
                self,
                duplicate_predicate=self._duplicate_predicate,
                duplicate_tolerance=self._duplicate_tolerance,
            )

        self.index.normals()

        return self

    def rasterize(
        self,
        dim: str,
        res: float,
        bbox: Optional[BoundingBoxLike] = None,
        meta: Optional[rasterio.profiles.Profile] = None,
    ) -> raster.Raster:
        """Rasterize a single point dimension from the standard, variable, or extended variable length point record of the point cloud.

        .. note::
            The dimension value at the cell centers of the resulting raster is interpolated using Laplace interpolation. Because this approach does not support extrapolation, any no-data cells remaining after the primary interpolation process are automatically identified and filled with ``rasterio.fill.fillnodata()``.

        :param dim: The point dimension to rasterize. See `Point Records <https://laspy.readthedocs.io/en/latest/intro.html#point-records>`_ for more information.

        :param res: The resolution of the resulting raster in the coordinate reference system of the point cloud.

        :param bbox: The two-dimensional, coordinate interleaved, axis-aligned bounding box, :math:`[x_{min}, y_{min}, x_{max}, y_{max}]`, specifying the spatial extent of the resulting raster in the coordinate reference system of the point cloud. Defaults to ``PointCloud.bbox`` if left unspecified.

        :param meta: The `profile <https://rasterio.readthedocs.io/en/stable/topics/profiles.html>`_ of the resulting raster. Defaults to :class:`raster.DefaultProfile` if left unspecified.

        :return: The resulting raster.
        """
        # Resolve the input.
        _bbox = bbox if bbox is not None else self.bbox

        # Initialize the index.
        # TODO: Factor this block out to a private method.
        if self.index is None:
            self._index = Index(
                self,
                duplicate_predicate=self._duplicate_predicate,
                duplicate_tolerance=self._duplicate_tolerance,
            )
        self.index.update(self.points[dim].scaled_array(), dim=dim)

        # Generate the output.
        out = raster.Raster(res, bbox=_bbox, meta=meta)
        with tqdm(desc="Rasterization", total=len(out), unit="cells") as pbar:
            for row in range(out.height):
                y = _bbox[3] - res * (row + 0.5)
                for col in range(out.width):
                    x = _bbox[0] + res * (col + 0.5)
                    out[row, col] = self._index.interpolate([x, y])
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

    def save(self, filepath: str | bytes | BytesIO) -> None:
        """Save the point cloud to  a `LASer (LAS)
        <https://www.asprs.org/divisions-committees/lidar-division/laser-las-file
        -format -exchange-activities>`_ or `LASzip <https://rapidlasso.de/laszip/>`_
        file.

        :param filepath: The path to the resulting file.
        """
        with laspy.open(filepath, mode="w", header=self.header) as dst:
            dst.write_points(self.data.points)


class Index:
    def __init__(
        self,
        pt_cloud: PointCloud,
        duplicate_predicate: Literal["First", "Highest", "Last", "Lowest"],
        duplicate_tolerance: float,
    ) -> None:
        # The underlying DT.
        self._dt = startinpy.DT(extra_attributes=True)
        self._dt.duplicates_handling = duplicate_predicate
        self._dt.snap_tolerance = duplicate_tolerance

        # The point dimension currently stored in the index.
        self._pt_dim = "z"

        for i, pt in tqdm(
            enumerate(pt_cloud.data.xyz),
            desc="Index Initialization",
            total=len(pt_cloud),
            unit="points",
        ):
            self._dt.insert_one_pt(pt, pid=i)

        # Allow the point dimension to be overwritten regardless of the currently
        # active duplicate handling method.
        self._dt.duplicates_handling = "Last"

    def __len__(self) -> int:
        return self._dt.number_of_vertices()

    @property
    def duplicates_handling(self) -> Literal["First", "Highest", "Last", "Lowest"]:
        return self._dt.duplicates_handling

    @duplicates_handling.setter
    def duplicates_handling(
        self, method: Literal["First", "Highest", "Last", "Lowest"]
    ) -> None:
        self._dt.duplicates_handling = method

    @property
    def snap_tolerance(self) -> float:
        return self._dt.snap_tolerance

    @snap_tolerance.setter
    def snap_tolerance(self, tol: float) -> None:
        self._dt.snap_tolerance = tol

    @property
    def point_dim(self) -> Optional[str]:
        return self._pt_dim

    @property
    def point_map(self):
        return self._dt.attribute("pid")[1:].astype(int)

    def interpolate(
        self,
        xy: Iterable[float],
        method: Literal["IDW", "Laplace", "NN", "NNI", "TIN"] = "Laplace",
    ) -> float:
        return self._dt.interpolate({"method": method}, [xy])

    def normals(self):
        def normal_triangle(tr, pts):
            v0 = pts[tr[1]] - pts[tr[0]]
            v1 = pts[tr[2]] - pts[tr[0]]
            n = np.cross(v0, v1)
            l = np.linalg.norm(n)
            return n / l

        pts = self._dt.points
        for pt_id in tqdm(range(1, len(pts))):
            incident_triangles = self._dt.incident_triangles_to_vertex(pt_id)
            incident_normals = np.empty([incident_triangles.shape[0], 3])
            for tri_id, tri_pt_ids in enumerate(incident_triangles):
                if self._dt.is_finite(tri_pt_ids):
                    tri_normal = normal_triangle(tri_pt_ids, pts)
                    incident_normals[tri_id] = tri_normal
            pt_normal = np.average(incident_normals, axis=0)
            pt_normal = pt_normal / np.linalg.norm(pt_normal)
            s = json.dumps({"nx": pt_normal[0], "ny": pt_normal[1], "nz": pt_normal[2]})
            self._dt.set_vertex_attributes(pt_id, s)

    def update(self, z: np.ndarray[tuple[Any,], np.dtype[np.number]], dim: str) -> None:
        if dim == self.point_dim:
            return

        self._pt_dim = dim

        for i, pid in tqdm(
            enumerate(self.point_map),
            desc="Index Update",
            total=len(self),
            unit="points",
        ):
            self._dt.update_vertex_z_value(i, z[pid])


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
    pc = PointCloud("small.laz")
    pc.normals()
    x = pc.index._dt.attribute("nx")
