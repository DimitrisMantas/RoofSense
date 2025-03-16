from __future__ import annotations

import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import rasterio
import rasterio.features
import rasterio.merge
from typing_extensions import override

from roofsense.bag3d import BAG3DTileStore
from roofsense.preprocessing.parsers.generic import (
    BAG3DTileAssetParser,
    BAG3DTileAssetParsingStage,
)
from roofsense.utilities import pcloud
from roofsense.utilities.file import confirm_write_op
from roofsense.utilities.pcloud import PointCloud
from roofsense.utilities.raster import DefaultProfile, Raster, get_raster_metadata


def parse_elevation(parser: LiDARParser, tile_id: str, overwrite: bool) -> None:
    dst_path = parser.resolve_filepath(tile_id + ".elev.tif")
    if confirm_write_op(dst_path, overwrite=overwrite):
        dsm = _rasterize_all_valid(
            parser.pntcl, field="z", res=parser.resol, bbox=parser.bbox
        )
        dsm.save(dst_path)
    else:
        # TODO: Load the DSM only if one or more of its derivatives need to be
        #  computed.
        src: rasterio.io.DatasetReader
        with rasterio.open(dst_path) as src:
            dsm = Raster(
                resol=parser.resol,
                bbox=parser.bbox,
                meta=DefaultProfile(dtype=src.dtypes[0], nodata=src.nodata),
            )
            # TODO: Check whether command makes a copy of the DSM.
            # If it does, move the raster initialization block outside of this
            # scope.
            dsm.data = src.read(indexes=1)

    _compute_ndrm(parser, tile_id, overwrite, dsm)
    _compute_slope(parser, tile_id, overwrite, dsm)


def parse_density(parser: LiDARParser, tile_id: str, overwrite: bool) -> None:
    dst_path = parser.resolve_filepath(tile_id + ".den.tif")
    if confirm_write_op(dst_path, overwrite=overwrite):
        parser.pntcl.density(parser.bbox).save(dst_path)


def parse_reflectance(parser: LiDARParser, tile_id: str, overwrite: bool) -> None:
    dst_path = parser.resolve_filepath(tile_id + ".rfl.tif")
    if not confirm_write_op(dst_path, overwrite=overwrite):
        return

    raster = _rasterize_all_valid(
        parser.pntcl, field="Reflectance", res=parser.resol, bbox=parser.bbox
    )
    # Convert dB to a linear scale to ensure the resulting raster can be scaled correctly.
    raster.data = 10 ** (0.1 * raster.data)
    # Reflectance values above 1 are considered to be noise because they correspond to non-Lambertian reflectors.
    raster.data = raster.data.clip(max=1)
    raster.save(dst_path)


def _compute_ndrm(
    parser: BAG3DTileAssetParser, tile_id: str, overwrite: bool, dsm: Raster
) -> None:
    dst_path = parser.resolve_filepath(tile_id + ".ndsm.tif")
    if not confirm_write_op(dst_path, overwrite=overwrite):
        return

    buildings = parser.surfaces.dissolve(by="identificatie")

    ras = deepcopy(dsm)
    ras.data -= rasterio.features.rasterize(
        shapes=(
            (building, med_roof_height)
            for building, med_roof_height in zip(
                buildings.geometry, buildings["b3_h_50p"]
            )
        ),
        transform=ras.transform,
        out_shape=ras.data.shape,
    )
    ras.save(dst_path)


def _compute_slope(
    parser: BAG3DTileAssetParser, tile_id: str, overwrite: bool, dsm: Raster
) -> None:
    dst_path = parser.resolve_filepath(tile_id + ".slp.tif")
    if confirm_write_op(dst_path, overwrite=overwrite):
        dsm.slope().save(dst_path)


# TODO: Add optional fill in ``pcloud.PointCloud,rasterize``.
def _rasterize_all_valid(
    pc: PointCloud, field: str, res: float, bbox: Sequence[float]
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


class LiDARParser(BAG3DTileAssetParser):
    """Parser for point clouds of the Dutch National Elevation Program"""

    def __init__(
        self,
        tile_store: BAG3DTileStore,
        callbacks: BAG3DTileAssetParsingStage
        | Iterable[BAG3DTileAssetParsingStage]
        | None = (parse_reflectance, parse_elevation, parse_density),
    ) -> None:
        super().__init__(tile_store, callbacks)

        # Extend the base parser to include additional stage arguments.
        self._bbox: Sequence[float] | None = None
        self._resol: float | None = None
        self._pntcl: PointCloud | None = None

    @property
    def bbox(self) -> Sequence[float] | None:
        return self._bbox

    @property
    def resol(self) -> float | None:
        return self._resol

    @property
    def pntcl(self) -> PointCloud | None:
        return self._pntcl

    @override
    def parse(self, tile_id: str, overwrite: bool = False) -> None:
        bbox, resol = get_raster_metadata(
            self.resolve_filepath(tile_id + ".rgb.tif"), "bounds", "res"
        )
        resol = 3 * resol[0]

        self._bbox = bbox
        self._resol = resol
        # This is technically also a parsing stage, but, because it's significantly faster to keep the point cloud in memory instead of continuously re-reading it from disk, the corresponding callback must return.
        # Hence, this callback is handled differently than the rest for the sake of clarity.
        # It is also possible to register it as a normal callback, but the attribute assignment will be external to the class.
        self._pntcl = self._parse_point_cloud(tile_id, overwrite, bbox)

        super().parse(tile_id, overwrite)

    def _parse_point_cloud(
        self, tile_id: str, overwrite: bool, bbox: Sequence[float]
    ) -> PointCloud:
        # TODO: Consider refactoring this block into a separate method.
        dst_path = self.resolve_filepath(tile_id + ".LAZ")
        if confirm_write_op(dst_path, overwrite=overwrite):
            src_paths = [
                self.resolve_filepath(id_ + ".LAZ") for id_ in self._manifest.lidar.tid
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
