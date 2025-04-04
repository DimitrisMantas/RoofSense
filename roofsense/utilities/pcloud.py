from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache
from io import BytesIO
from os import PathLike
from typing import Self

import laspy
import numpy as np
import polars as pl
import rasterio
import scipy as sp
import tqdm

from roofsense.utilities import raster
from roofsense.utilities.raster import DefaultProfile
from roofsense.utilities.types import BoundingBoxLike


class PointCloud:
    """Convenience class for manipulating point clouds in the `LASer
    <https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format
    -exchange-activities>`_ and `LASzip <https://rapidlasso.de/laszip/>`_ file
    formats.
    """

    def __init__(
        self, filepath: str | bytes | BytesIO, init_index: bool = False
    ) -> None:
        """Initialize the point cloud.

        :param filepath: The path to the file containing the point cloud.

        :param init_index: A boolean flag indicating whether to index the point cloud
        upon its initialization. Regardless of the value of this flag, a spatial
        index is created automatically, if one does not already exist, to support all
        operations whose performance would significantly benefit from optimized
        spatial access methods.
        """
        # Initialize the cloud.
        with laspy.open(filepath) as src:
            self._data = src.read()

        # Initialize the index.
        self._index = Index(self) if init_index else None

    @property
    def data(self):
        return self._data

    # TODO: Document this method.
    def __getattr__(self, item):
        return getattr(self.data, item)

    # TODO: Document this method.
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        """Get the number of points in the point cloud."""
        return len(self.data)

    @property
    def bbox(self) -> BoundingBoxLike:
        """Get the two-dimensional, coordinate interleaved, axis-aligned bounding
        box, :math:`[x_{min}, y_{min}, x_{max}, y_{max}]`, specifying the spatial
        extent of the point cloud in the length unit of its coordinate reference
        system.
        """
        return *self.header.mins[:2], *self.header.maxs[:2]

    @property
    def index(self) -> Index | None:
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

        self._data.points = self.points[
            np.logical_and(
                np.logical_and(xmin <= self.X, self.X <= xmax),
                np.logical_and(ymin <= self.Y, self.Y <= ymax),
            )
        ]

        # Invalidate the index.
        if self.index is not None:
            self._index = Index(self)

        return self

    def normals(self) -> Self:
        # Initialize the index.
        if self.index is None:
            self._index = Index(self)

        raise NotImplementedError

    def rasterize(
        self,
        dim: str,
        res: float,
        p: float = 2,
        bbox: BoundingBoxLike | None = None,
        meta: rasterio.profiles.Profile | None = None,
    ) -> raster.Raster:
        """Rasterize a single point dimension from the standard, variable,
        or extended variable length point record of the point cloud.

        .. note::
            The dimension value at the cell centers of the resulting raster is
            interpolated using Laplace interpolation. Because this approach does not
            support extrapolation, any no-data cells remaining after the primary
            interpolation process are automatically identified and filled with
            ``rasterio.fill.fillnodata()``.

        :param dim: The point dimension to rasterize. See `Point Records
        <https://laspy.readthedocs.io/en/latest/intro.html#point-records>`_ for more
        information.

        :param res: The resolution of the resulting raster in the coordinate
        reference system of the point cloud.

        :param bbox: The two-dimensional, coordinate interleaved, axis-aligned
        bounding box, :math:`[x_{min}, y_{min}, x_{max}, y_{max}]`, specifying the
        spatial extent of the resulting raster in the coordinate reference system of
        the point cloud. Defaults to ``PointCloud.bbox`` if left unspecified.

        :param meta: The `profile <https://rasterio.readthedocs.io/en/stable/topics
        /profiles.html>`_ of the resulting raster. Defaults to
        :class:`raster.DefaultProfile` if left unspecified.

        :return: The resulting raster.
        """
        # Compute the scaled dimension.
        attr = self[dim].scaled_array()

        # Disambiguate the input.
        bbox = bbox if bbox is not None else self.bbox

        # Initialize the index.
        # TODO: Factor this block out to a private method.
        if self.index is None:
            self._index = Index(self)

        # Initialize the output raster.
        ras = raster.Raster(
            res,
            bbox=bbox,
            meta=meta
            if meta is not None
            else DefaultProfile(
                crs=self.header.parse_crs().to_string(),
                dtype=np.float32
                if np.issubdtype(attr.dtype, np.floating)
                else attr.dtype,
                nodata=np.nan
                if np.issubdtype(attr.dtype, np.floating)
                else np.iinfo(attr.dtype).max,
            ),
        )
        cells = ras.xy()

        # Query the index.
        neighbors, distances = self.index.query(
            tuple(map(tuple, cells)),
            r=res / 2,
            # Chebyshev Distance
            p=np.inf,
        )

        # TODO: Add multithreading.
        # Interpolate the attribute value at the raster cells.
        ras_data = np.full_like(
            neighbors, fill_value=ras.meta["nodata"], dtype=ras.meta["dtype"]
        )
        for i, (neighbor, distance) in tqdm.tqdm(
            enumerate(zip(neighbors, distances)),
            desc="Rasterization",
            total=len(neighbors),
            unit="cells",
        ):
            if not neighbor:
                # The cell is empty.
                continue
            if np.any(distance == 0):
                # The cell is a member of the point cloud; read the corresponding
                # attribute value.
                ras_data[i] = attr[neighbor[np.argsort(distance)[0]]]
            else:
                # IDW
                ras_data[i] = np.average(attr[neighbor], weights=distance**-p)

        # Reshape the raster data into a two-dimensional array.
        ras_data = ras_data.reshape([ras.height, ras.width])

        # Overwrite the raster data.
        ras.data = ras_data

        return ras

    def density(
        self,
        bbox: BoundingBoxLike | None = None,
        meta: rasterio.profiles.Profile | None = None,
    ) -> float | raster.Raster:
        """Compute the planar density of the point cloud.

        Args:
            res: The spatial resolution of the output raster in the CRS of the point cloud. If provided, the density is computed per cell. Otherwise, a global density is returned.
        """
        # Disambiguate the input.
        bbox = bbox if bbox is not None else self.bbox

        # if res is None:
        #     area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        #     return len(self) / area

        return self._rasterize_density(bbox, meta)

    def _rasterize_density(
        self,
        bbox: BoundingBoxLike | None = None,
        meta: rasterio.profiles.Profile | None = None,
    ) -> raster.Raster:
        # Initialize the index.
        # TODO: Factor this block out to a private method.
        if self.index is None:
            self._index = Index(self)

        # Initialize the output raster.
        ras = raster.Raster(
            resol=1,
            bbox=bbox,
            meta=meta
            if meta is not None
            else DefaultProfile(
                crs=self.header.parse_crs().to_string(),
                dtype=np.uint16,
                nodata=np.iinfo(np.uint16).max,
            ),
        )
        cells = ras.xy()

        # Query the index.
        num_neighbors = self.index.query(
            tuple(map(tuple, cells)),
            r=0.5,
            # Chebyshev Distance
            p=np.inf,
            return_distances=False,
            return_length=True,
        )

        ras.data = num_neighbors.reshape([ras.height, ras.width])

        return ras

    def remove_duplicates(self) -> Self:
        # Gather the point records.
        # NOTE: The unique element filter provided by NumPy is inefficient.
        #       See https://github.com/numpy/numpy/issues/11136 for more information.
        recs = pl.DataFrame(
            np.vstack([self.X, self.Y, self.Z]).transpose(), schema=["X", "Y", "Z"]
        )
        recs = recs.with_row_index()

        # Sort the point records.
        # NOTE: This ensures that only contextually irrelevant points are discarded.
        recs = recs.sort("Z")

        # Filter the point records.
        # NOTE: The unique point indices must be sorted so as to not disturb the
        #       spatial coherence of the point cloud.
        self._data.points = self.data.points[
            recs.unique(subset=["X", "Y"], keep="last")["index"].sort().to_numpy()
        ]

        return self

    def save(self, filepath: str | bytes | BytesIO) -> None:
        """Save the point cloud to a `LASer
        <https://www.asprs.org/divisions-committees/lidar-division/laser-las-file
        -format-exchange-activities>`_ and `LASzip <https://rapidlasso.de/laszip/>`_
        file.

        :param filepath: The path to the output file.
        """
        with laspy.open(filepath, mode="w", header=self.header) as dst:
            dst.write_points(self.data.points)


class Index:
    def __init__(self, pc: PointCloud, workers: int = -1, **kwargs) -> None:
        self.workers = workers

        self._struct = sp.spatial.KDTree(
            np.vstack([pc.data.x, pc.data.y]).transpose(), **kwargs
        )

    def __len__(self) -> int:
        return self._struct.n

    def normals(self):
        raise NotImplementedError

    # TODO: Add support for non-hashable parameters.
    # TODO: Harmonize the output data formats.
    # noinspection PyProtectedMember
    @cache
    def query(
        self,
        x: np._typing._array_like._ArrayLikeFloat_co,
        k: int | Iterable[int] | None = None,
        r: float | Iterable[float] | None = None,
        return_distances: bool = True,
        **kwargs,
    ):
        common_args = {"x": x, "workers": self.workers}
        if k is not None and r is None:
            distances, neighbors = self._struct.query(k=k, **common_args, **kwargs)
        elif k is None and r is not None:
            neighbors, distances = self._query_radius(
                r=r, return_distances=return_distances, **common_args, **kwargs
            )
        else:
            raise ValueError(
                f"Could not disambiguate query type. Found non-null values for both "
                f"{k!r} and {r!r}."
            )

        if return_distances:
            return neighbors, distances
        else:
            return neighbors

    # noinspection PyProtectedMember
    def _query_radius(
        self,
        x: np._typing._array_like._ArrayLikeFloat_co,
        r: float | Iterable[float],
        return_distances: bool,
        **kwargs,
    ):
        x = np.asarray(x)

        # Coerce the array to 2D.
        if x.ndim == 1:
            x = x[None, :]

        # Find the neighbors.
        neighbors = self._struct.query_ball_point(
            x,
            r=r,
            # NOTE: Multi-point queries are sorted by index, which is not necessary.
            return_sorted=False,
            **kwargs,
        )

        if not return_distances:
            return neighbors, None

        # Find the distances.
        distances = np.empty_like(neighbors)
        for i, (cell, neighbor) in enumerate(zip(x, neighbors)):
            # NOTE: Distance subarrays must be one-dimensional to match the neighbor
            # array.
            distances[i] = sp.spatial.distance.cdist(
                cell[None, :], self._struct.data[neighbor]
            ).ravel()

        return neighbors, distances


def merge(
    ipaths: Sequence[str | PathLike],
    opath: str | PathLike,
    crop: BoundingBoxLike | None = None,
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
