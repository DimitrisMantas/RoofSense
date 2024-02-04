from __future__ import annotations

import rasterio.merge
from typing_extensions import override

import config
import utils
from parsers.base import AssetParser
from parsers.utils import raster


class ImageParser(AssetParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        self._update(obj_id)

        cir_paths = [
            f"{config.env('TEMP_DIR')}{'CIR'}_{img_id}"
            for img_id in self._manifest["image_ids"]
        ]
        rgb_paths = [
            f"{config.env('TEMP_DIR')}{'RGB'}_{img_id}"
            for img_id in self._manifest["image_ids"]
        ]
        # TODO: Parallelize this operation.
        # TODO: Read the object ID from the asset manifest.
        self._parse_cir_assets(obj_id, cir_paths)
        self._parse_rgb_assets(obj_id, rgb_paths)

    def _parse_cir_assets(self, obj_id: str, paths: list[str]) -> None:
        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('NIR_EXTENSION')}"
            f"{config.var('TIFF')}"
        )
        if utils.file.exists(out_path):
            return
        # NOTE: Mapping the output image to an integer-coordinate grid ensures that it
        #       will be aligned pixel-wise with its constituents.
        rasterio.merge.merge(
            paths,
            bounds=self._surfs.total_bounds.tolist(),
            target_aligned_pixels=True,
            dst_path=out_path,
            dst_kwds=raster.SingleBandProfile(),
            indexes=[1],
        )

    def _parse_rgb_assets(self, obj_id: str, paths: list[str]) -> None:
        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RGB_EXTENSION')}"
            f"{config.var('TIFF')}"
        )
        if utils.file.exists(out_path):
            return
        rasterio.merge.merge(
            paths,
            bounds=self._surfs.total_bounds.tolist(),
            target_aligned_pixels=True,
            dst_path=out_path,
            dst_kwds=raster.MultiBandProfile(),
        )
