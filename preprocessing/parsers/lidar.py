from __future__ import annotations

import os

import rasterio
from overrides import override

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
        rfl_path = f"{config.env('TEMP_DIR')}{obj_id}.rfl{config.var('TIF')}"
        slp_path = f"{config.env('TEMP_DIR')}{obj_id}.slp{config.var('TIF')}"
        scalars = _get_scalars(rfl_path, slp_path)
        if not scalars:
            return
        rasters={
            scalar:pcloud.PointCloud(ldr_path).rasterize(scalar,res=float(config.var("RESOLUTION")),bbox=_get_bbox(obj_id)) for scalar in scalars
        }
        refl_field = config.var("REFLECTANCE_FIELD")
        elev_field = config.var("ELEVATION_FIELD")
        if refl_field in scalars:
            rasters[refl_field].save(rfl_path)
        if elev_field in scalars:
            rasters[elev_field].slope().save(slp_path)

    def _merge_assets(self, obj_id: str) -> str:
        out_path = f"{config.env('TEMP_DIR')}{obj_id}{config.var('LAZ')}"
        if not utils.file.exists(out_path):
            in_paths = [os.path.join(config.env('TEMP_DIR'), f"{ldr_id}.LAZ") for ldr_id
                in self._manifest["lidar"]["tid"]]
            pcloud.merge(
                in_paths,
                out_path,
                crop=self._surfs.total_bounds,
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
