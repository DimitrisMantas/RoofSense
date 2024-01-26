from __future__ import annotations

import json
import pathlib

import rasterio.merge
from typing_extensions import override

import config
import utils
from parsers.base import DataParser
from parsers.utils import raster


class ImageDataParser(DataParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        self._update_fields(obj_id)

        cir_paths = [
            f"{config.env('TEMP_DIR')}{'CIR'}_{img_id}"
            for img_id in self._data["image_ids"]
        ]
        rgb_paths = [
            f"{config.env('TEMP_DIR')}{'RGB'}_{img_id}"
            for img_id in self._data["image_ids"]
        ]

        # FIXME: Merging images results in the result product being shifted in
        #        relation to its constituents.
        # TODO: Parallelize this operation.
        # TODO: Read the object ID from the asset manifest.
        self._parse_cir_images(obj_id, cir_paths)
        self._parse_rgb_images(obj_id, rgb_paths)

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

    def _parse_cir_images(self, obj_id: str, paths: list[str]):
        path = f"{config.env('TEMP_DIR')}{obj_id}.nir{config.var('TIFF')}"
        if utils.file.exists(path):
            return
        merged_img, merged_transform = rasterio.merge.merge(
            paths, bounds=self._surfs.total_bounds.tolist(), indexes=[1]
        )
        raster.write(merged_img, merged_transform, path)

    def _parse_rgb_images(self, obj_id: str, paths: list[str]):
        path = f"{config.env('TEMP_DIR')}{obj_id}.rgb{config.var('TIFF')}"
        if utils.file.exists(path):
            return
        merged_img, merged_transform = rasterio.merge.merge(
            paths, bounds=self._surfs.total_bounds.tolist()
        )
        raster.write(merged_img, merged_transform, path)
