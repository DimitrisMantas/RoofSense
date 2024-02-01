from __future__ import annotations

import json
import pathlib
import time

import rasterio
from typing_extensions import override

import config
import utils
from parsers.base import DataParser
from parsers.utils import pcloud


class LiDARDataParser(DataParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        self._update_fields(obj_id)

        ipaths = [
            f"{config.env('TEMP_DIR')}{ldr_id}" for ldr_id in self._data["lidar_ids"]
        ]
        opath = f"{config.env('TEMP_DIR')}{obj_id}{config.var('LAZ')}"
        ref_img_path = f"{config.env('TEMP_DIR')}{obj_id}.nir{config.var('TIFF')}"
        # NOTE: something about the remove duplicate thing
        if not utils.file.exists(opath):
            t0 = time.perf_counter()
            pcloud.merge(ipaths, opath, crop=self._surfs.total_bounds, rem_dpls=True)
            print(time.perf_counter() - t0)
        f: rasterio.io.DatasetReader
        with rasterio.open(ref_img_path) as f:
            ref_meta = f.profile
        left = ref_meta["transform"].c
        top = ref_meta["transform"].f

        right = ref_meta["transform"].a * ref_meta["width"] + left
        bottom = ref_meta["transform"].e * ref_meta["height"] + top
        ref_bbox = [left, bottom, right, top]
        rasters = pcloud.PointCloud(opath).rasterize(
            ["z", "Reflectance"], resol=0.25, bbox=ref_bbox
        )
        slope_opath = f"{config.env('TEMP_DIR')}{obj_id}.slp{config.var('TIFF')}"
        refl_opath = f"{config.env('TEMP_DIR')}{obj_id}.rfl{config.var('TIFF')}"
        rasters["z"].slope().save(slope_opath)
        rasters["Reflectance"].save(refl_opath)

    # FIXME: Write an asset data parser to avoid code duplication
    @override
    def _update_fields(self, obj_id: str) -> None:
        path = (
            f"{config.var('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('ASSET_MANIFEST_EXTENSION')}"
            f"{config.var('JSON')}"
        )
        with pathlib.Path(path).open() as f:
            self._data = json.load(f)

        self._surfs = utils.geom.buffer(utils.geom.read_surfaces(obj_id))
