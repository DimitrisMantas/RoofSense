from __future__ import annotations

import rasterio.merge
from typing_extensions import override

import config
import utils
from preprocessing.parsers.base import AssetParser
from preprocessing.parsers.utils import raster


class ImageParser(AssetParser):
    def __init__(self) -> None:
        super().__init__()

    @override
    def parse(self, obj_id: str) -> None:
        self._update(obj_id)

        # TODO: Parallelize this operation.
        self._parse_cir_images(obj_id)
        self._parse_rgb_images(obj_id)

    def _parse_cir_images(self, obj_id: str) -> None:
        in_paths = [
            f"{config.env('TEMP_DIR')}{'CIR'}_{img_id}"
            for img_id in self._manifest[config.var("ASSET_MANIFEST_IMAGE_IDS")]
        ]
        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('NIR')}"
            f"{config.var('TIFF')}"
        )
        if utils.file.exists(out_path):
            return
        rasterio.merge.merge(
            in_paths,
            bounds=self._surfs.total_bounds.tolist(),
            # Map the output image to an integer-coordinate grid.
            # NOTE: This ensures that the image will be aligned pixel-wise with its
            #       constituents.
            target_aligned_pixels=True,
            dst_path=out_path,
            dst_kwds=raster.SingleBandProfile(),
            indexes=[1],
        )

    def _parse_rgb_images(self, obj_id: str) -> None:
        in_paths = [
            f"{config.env('TEMP_DIR')}{'RGB'}_{img_id}"
            for img_id in self._manifest[config.var("ASSET_MANIFEST_IMAGE_IDS")]
        ]
        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RGB')}"
            f"{config.var('TIFF')}"
        )
        if utils.file.exists(out_path):
            return
        rasterio.merge.merge(
            in_paths,
            bounds=self._surfs.total_bounds.tolist(),
            target_aligned_pixels=True,
            dst_path=out_path,
            dst_kwds=raster.MultiBandProfile(),
        )
