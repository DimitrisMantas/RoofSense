import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import rasterio
import rasterio.features
import rasterio.merge
from typing_extensions import override

from roofsense.parsers.base import AssetParser
from roofsense.utils import pcloud
from roofsense.utils.file import confirm_write_op
from roofsense.utils.pcloud import PointCloud
from roofsense.utils.raster import DefaultProfile, Raster


class LiDARParser(AssetParser):
    """AHN4 Tile Parser."""

    def __init__(self, dirpath: str) -> None:
        # TODO: Check whether the initializer documentation can be copied from the
        #  parent class.
        """Configure the parser.

        Args:
            dirpath:
                The path to the data directory.
        """
        super().__init__(dirpath)

    @override
    def parse(self, tile_id: str, overwrite: bool = False) -> None:
        """Parse the AHN4 data corresponding to a particular 3DBAG tile.

        This method first merges the input data to the tile bounds and in turn
        rasterizes the reflectance and elevation fields as well as the planar point
        density of the output cloud.
        The spatial resolution of the resulting rasters is three times that of the
        respective BM5 data (i.e., 3 × 8 cm = 24 cm) and share the same bounds.
        See ``roofsense.utils.pcloud.merge`` and
        ``roofsense.utils.pcloud.PointCloud.rasterize`` for more implementation
        details on merging and point cloud rasterization, respectively.

        Once rasterized, reflectance values are rescaled from decibels (i.e.,
        logarithmic scale) to the underlying optical power ratio (i.e., linear scale).
        This transformation enables bilinear interpolation of the output to a
        different resolution.

        In addition, values which correspond to non-Lambertian reflectors (i.e.,
        ≥ 1) are discarded by clipping the resulting data to [0, 1].
        This correction is performed because although certain materials of interest (
        e.g., glass, metal, etc.) can behave as specular reflectors under certain
        lightning or viewing conditions, their exceedingly high signal can overpower
        that of neighboring pixels in the context of convolution and pooling operations.
        Furthermore, interpolation to or from these values may result in erroneous
        intermediates being introduced.

        Finally, the elevation raster (DSM) is used to compute the nDRM and slope of
        the underlying scene.

        See ``rasterio.features.rasterize`` and
        ``roofsense.utils.raster.Raster.slope`` for more implementation details on
        vector data rasterization and slope calculations on rasters, respectively.

        Warnings:
            This method requires that the BM5 data corresponding to the provided tile
            has already been parsed and that the respective output image exists in
            the specified data directory.

        Args:
            tile_id:
                The tile ID.
            overwrite:
                A flag indicating whether to overwrite any previous output.
        """
        self._update(tile_id)

        res, bbox = self._query_image_meta(tile_id, "res", "bounds")
        res = 3 * res[0]

        pc = self._merge_tiles(tile_id, overwrite, bbox)

        self._parse_reflectance(tile_id, overwrite, pc, res, bbox)
        self._parse_elevation_and_compute_derivatives(tile_id, overwrite, pc, res, bbox)
        self._parse_density(tile_id, overwrite, pc, bbox)

    # TODO: Refactor this method as a function in a tile utility module.
    def _query_image_meta(self, tile_id: str, attr: str, *attrs: str) -> list[Any]:
        src_path = self.resolve_filepath(tile_id + ".rgb.tif")
        src: rasterio.io.DatasetReader
        with rasterio.open(src_path) as src:
            return [getattr(src, attr) for attr in [attr] + list(attrs)]

    def _merge_tiles(
        self, tile_id: str, overwrite: bool, bbox: Sequence[float]
    ) -> PointCloud:
        # TODO: Consider refactoring this block into a separate method.
        dst_path = self.resolve_filepath(tile_id + ".LAZ")
        if confirm_write_op(dst_path, overwrite=overwrite):
            src_paths = [
                self.resolve_filepath(id_ + ".LAZ")
                for id_ in self._manifest["lidar"]["tid"]
            ]
            pcloud.merge(
                src_paths,
                dst_path,
                crop=bbox,
                # NOTE: The AHN4 tiles served by GeoTiles have a 20 m overlap with
                # each other.
                # See https://weblog.fwrite.org/kaartbladen/ for more information.
                rem_dpls=True,
            )
        return PointCloud(dst_path)

    def _parse_reflectance(
        self,
        tile_id: str,
        overwrite: bool,
        pc: PointCloud,
        res: float,
        bbox: Sequence[float],
    ) -> None:
        dst_path = self.resolve_filepath(tile_id + ".rfl.tif")
        if not confirm_write_op(dst_path, overwrite=overwrite):
            return

        ras = self._rasterize_all_valid(pc, field="Reflectance", res=res, bbox=bbox)
        ras.data = (10 ** (0.1 * ras.data)).clip(max=1)
        ras.save(dst_path)

    def _parse_elevation_and_compute_derivatives(
        self,
        tile_id: str,
        overwrite: bool,
        pc: PointCloud,
        res: float,
        bbox: Sequence[float],
    ) -> None:
        dst_path = self.resolve_filepath(tile_id + ".elev.tif")
        if confirm_write_op(dst_path, overwrite=overwrite):
            ras = self._rasterize_all_valid(pc, field="z", res=res, bbox=bbox)
            ras.save(dst_path)
        else:
            # TODO: Load the DSM only if one or more of its derivatives need to be
            #  computed.
            src: rasterio.io.DatasetReader
            with rasterio.open(dst_path) as src:
                ras = Raster(
                    resol=res,
                    bbox=bbox,
                    meta=DefaultProfile(dtype=src.dtypes[0], nodata=src.nodata),
                )
                # TODO: Check whether command makes a copy of the DSM.
                # If it does, move the raster initialization block outside of this
                # scope.
                ras.data = src.read(indexes=1)

        self._compute_ndrm(tile_id, overwrite, ras)
        self._compute_slope(tile_id, overwrite, ras)

    def _compute_ndrm(self, tile_id: str, overwrite: bool, dsm: Raster) -> None:
        dst_path = self.resolve_filepath(tile_id + ".ndsm.tif")
        if not confirm_write_op(dst_path, overwrite=overwrite):
            return

        buildings = self._surfaces.dissolve(by="id")

        ras = deepcopy(dsm)
        ras.data -= rasterio.features.rasterize(
            shapes=(
                (building, med_roof_height)
                for building, med_roof_height in zip(
                    buildings.geometry, buildings["b3_h_dak_50p"]
                )
            ),
            transform=ras.transform,
            out_shape=ras.data.shape,
        )
        ras.save(dst_path)

    def _compute_slope(self, tile_id: str, overwrite: bool, dsm: Raster) -> None:
        dst_path = self.resolve_filepath(tile_id + ".slp.tif")
        if confirm_write_op(dst_path, overwrite=overwrite):
            dsm.slope().save(dst_path)

    def _parse_density(
        self, tile_id: str, overwrite: bool, pc: PointCloud, bbox: Sequence[float]
    ) -> None:
        dst_path = self.resolve_filepath(tile_id + ".den.tif")
        if confirm_write_op(dst_path, overwrite=overwrite):
            pc.density(bbox).save(dst_path)

    # TODO: Add optional fill in ``pcloud.PointCloud,rasterize``.
    def _rasterize_all_valid(
        self, pc: PointCloud, field: str, res: float, bbox: Sequence[float]
    ) -> Raster:
        ras = pc.rasterize(field, res, bbox=bbox)
        while (num_invalid := np.count_nonzero(~np.isfinite(ras.data))) != 0:
            msg = (
                f"Encountered {num_invalid} invalid (NaN or ±∞) values in output"
                f"raster."
                f" "
                f"Filling until all valid..."
            )
            warnings.warn(msg, RuntimeWarning)
            ras.fill()
        return ras
