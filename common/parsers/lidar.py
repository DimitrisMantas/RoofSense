from __future__ import annotations

import os
import warnings

import numpy as np
import rasterio
from typing_extensions import override

import config
import utils
from common.parsers.base import AssetParser
from utils import pcloud
from utils.type import BoundingBoxLike


class LiDARParser(AssetParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        self._update(obj_id)
        box = _get_bbox(obj_id)
        ldr_path = self._merge_assets(obj_id, box)
        rfl_path = f"{config.env('TEMP_DIR')}{obj_id}.rfl{config.var('TIF')}"
        slp_path = f"{config.env('TEMP_DIR')}{obj_id}.slp{config.var('TIF')}"
        #density
        den_path = f"{config.env('TEMP_DIR')}{obj_id}.den{config.var('TIF')}"
        scalars = _get_scalars(rfl_path, slp_path)
        if not scalars:
            return
        pc = pcloud.PointCloud(ldr_path)
        # TODO: optimize the resolution factor
        res = _get_res(obj_id) * 3

        refl_field = config.var("REFLECTANCE_FIELD")
        elev_field = config.var("ELEVATION_FIELD")

        if refl_field in scalars:
            rfl = pc.rasterize(refl_field, res=res, bbox=box)

            while (num_nodata := np.count_nonzero(np.isnan(rfl.data))) != 0:
                # Fill until the raster is valid.
                msg = f"Detected {num_nodata} no-data cells while processing {refl_field.lower()} raster. Filling until valid..."
                warnings.warn(msg, EncodingWarning)

                rfl.fill()

            # Overwrite with the underlying ratio.
            # NOTE: This rescaling allows the raster to be resampled using linear
            # interpolation.
            rfl.data = 10 ** (0.1 * rfl.data)
            # Discard values corresponding to non-Lambertian reflectors.
            # NOTE: Although certain materials of interest (e.g., glass) may appear
            # as specular reflectors under the appropriate conditions, they are
            # rarely encountered in practice, and their exceedingly high signal may
            # overpower that of neighboring areas.
            rfl.data = rfl.data.clip(max=1)

            rfl.save(rfl_path)
        if elev_field in scalars:
            elev = pc.rasterize(elev_field, res=res, bbox=box).slope()

            while (num_nodata := np.count_nonzero(np.isnan(elev.data))) != 0:
                # Fill until the raster is valid.
                msg = f"Detected {num_nodata} no-data cells while processing {refl_field.lower()} raster. Filling until valid..."
                warnings.warn(msg, EncodingWarning)

                elev.fill()

            elev.save(slp_path)

        # # rasterize density
        # if not os.path.isfile(den_path):
        #     den = pc.density(res=res, bbox=box)
        #
        #     while num_nodata := np.count_nonzero(np.isnan(den.data)) != 0:
        #         # Fill until the raster is valid.
        #         msg = f"Detected {num_nodata} no-data cells while processing {refl_field.lower()} raster. Filling until valid..."
        #         warnings.warn(msg, EncodingWarning)
        #
        #         den.fill()
        #
        #     den.save(den_path)

    def _merge_assets(self, obj_id: str, box) -> str:
        out_path = f"{config.env('TEMP_DIR')}{obj_id}{config.var('LAZ')}"
        if not utils.file.exists(out_path):
            in_paths = [
                os.path.join(config.env("TEMP_DIR"), f"{ldr_id}.LAZ")
                for ldr_id in self._manifest["lidar"]["tid"]
            ]
            pcloud.merge(
                in_paths,
                out_path,
                crop=box,
                # NOTE: The AHN4 tiles served by GeoTiles have a 20 m overlap with
                #       each other.
                rem_dpls=True,
            )
        return out_path


def _get_scalars(rfl_path: str, slp_path: str) -> list[str]:
    ids = []
    for scalar, path in zip(
        [config.var("REFLECTANCE_FIELD"), config.var("ELEVATION_FIELD")],
        [rfl_path, slp_path],
    ):
        if not utils.file.exists(path):
            ids.append(scalar)
    return ids


def _get_bbox(obj_id: str) -> BoundingBoxLike:
    img_path = (
        f"{config.env('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('RGB')}"
        f"{config.var('TIF')}"
    )
    f: rasterio.io.DatasetReader
    with rasterio.open(img_path) as f:
        ref_meta = f.profile
    xmin = ref_meta["transform"].c
    ymax = ref_meta["transform"].f
    xmax = ref_meta["transform"].a * ref_meta["width"] + xmin
    ymin = ref_meta["transform"].e * ref_meta["height"] + ymax
    return xmin, ymin, xmax, ymax


def _get_res(obj_id: str) -> float:
    img_path = (
        f"{config.env('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('RGB')}"
        f"{config.var('TIF')}"
    )
    f: rasterio.io.DatasetReader
    with rasterio.open(img_path) as f:
        return f.transform.a
