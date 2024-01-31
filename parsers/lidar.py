from __future__ import annotations

import json
import pathlib

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
        opath = f"{config.env('TEMP_DIR')}{obj_id}.ldr{config.var('LAZ')}"

        # NOTE: something about the remove duplicate thing
        pcloud.merge(
            ipaths, opath, crop=self._surfs.total_bounds, remove_duplicates=True
        )

        # TODO: Overwrite the bounding box with that of an image??
        rasters = pcloud.PointCloud(opath).rasterize(["z", "Reflectance"], resol=0.25)
        slope_opath = f"{config.env('TEMP_DIR')}{obj_id}.slp{config.var('TIFF')}"
        refl_opath = f"{config.env('TEMP_DIR')}{obj_id}.rfl{config.var('TIFF')}"
        rasters["z"].slope().save(slope_opath)
        rasters["Reflectance"].save(refl_opath)

    # FIXME: Write an asset data parser to avoid duplicate methods in the parser module.
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
