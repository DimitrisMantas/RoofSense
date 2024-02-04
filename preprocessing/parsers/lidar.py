from __future__ import annotations

import rasterio
from typing_extensions import override

import config
import utils
from preprocessing.parsers.base import AssetParser
from preprocessing.parsers.utils import pcloud
from utils.type import BoundingBoxLike


class LiDARParser(AssetParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        self._update(obj_id)

        ldr_path = self._merge_assets(obj_id)

        rfl_path = f"{config.env('TEMP_DIR')}{obj_id}.rfl{config.var('TIFF')}"
        slp_path = f"{config.env('TEMP_DIR')}{obj_id}.slp{config.var('TIFF')}"
        scalars = _get_scalar_ids(rfl_path, slp_path)
        if not scalars:
            return
        rasters = pcloud.PointCloud(ldr_path).rasterize(
            scalars, resol=0.25, bbox=_get_image_bbox(obj_id)
        )

        refl_field = config.var("REFLECTANCE_FIELD")
        elev_field = config.var("ELEVATION_FIELD")
        if refl_field in scalars:
            rasters[refl_field].save(rfl_path)
        if elev_field in scalars:
            rasters[elev_field].slope().save(slp_path)

    def _merge_assets(self, obj_id: str) -> str:
        in_paths = [
            f"{config.env('TEMP_DIR')}{ldr_id}"
            for ldr_id in self._manifest["lidar_ids"]
        ]
        out_path = f"{config.env('TEMP_DIR')}{obj_id}{config.var('LAZ')}"
        if not utils.file.exists(out_path):
            pcloud.merge(
                in_paths, out_path, crop=self._surfs.total_bounds, rem_dpls=True
            )
        return out_path


def _get_scalar_ids(rfl_path: str, slp_path: str) -> list[str]:
    ids = []
    for scalar, path in zip(
        [config.var("REFLECTANCE_FIELD"), config.var("ELEVATION_FIELD")],
        [rfl_path, slp_path],
    ):
        if not utils.file.exists(path):
            ids.append(scalar)
    return ids


def _get_image_bbox(obj_id: str) -> BoundingBoxLike:
    img_path = (
        f"{config.env('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('NIR_EXTENSION')}"
        f"{config.var('TIFF')}"
    )
    f: rasterio.io.DatasetReader
    with rasterio.open(img_path) as f:
        ref_meta = f.profile

    xmin = ref_meta["transform"].c
    ymax = ref_meta["transform"].f
    xmax = ref_meta["transform"].a * ref_meta["width"] + xmin
    ymin = ref_meta["transform"].e * ref_meta["height"] + ymax
    return xmin, ymin, xmax, ymax
