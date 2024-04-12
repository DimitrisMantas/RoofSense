from __future__ import annotations

import os.path

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

        in_paths = [
            os.path.join(config.env("TEMP_DIR"), img_id+".tif")
            for img_id in self._manifest["image"]["tid"]
        ]

        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RGB')}"
            f"{config.var('TIF')}"
        )

        if utils.file.exists(out_path):
            return

        rasterio.merge.merge(
            in_paths,
            bounds=self._surfs.total_bounds.tolist(),
            # Map the output image to an integer-coordinate grid.
            # NOTE: This ensures that the image is aligned pixel-wise with its
            #       constituents.
            target_aligned_pixels=True,
            dst_path=out_path,
            dst_kwds=raster.MultiBandProfile(),
        )
