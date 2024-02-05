from __future__ import annotations

import copy
import json
import pathlib

import rasterio

import config
import utils.file


class IRISConfigurationFile:
    def __init__(self) -> None:
        with pathlib.Path(config.var("IRIS_BASE_CONFIG_FILENAME")).open() as f:
            self._base = json.load(f)

    def create(self, obj_id: str) -> None:
        img_path = f"{config.env('TEMP_DIR')}{obj_id}.stack{'.tif'}"
        out_path = f"{config.env('TEMP_DIR')}{obj_id}.iris{'.json'}"
        if utils.file.exists(out_path):
            return
        f: rasterio.io.DatasetReader
        g: rasterio.io.DatasetWriter
        with rasterio.open(img_path) as f:
            # NOTE: IRIS expects raster shapes in reverse NumPy order.
            img_shape = [f.width, f.height]
        out = copy.deepcopy(self._base)
        out["images"]["shape"] = img_shape
        out["segmentation"]["mask_area"] = [0, 0, *img_shape]
        with open(out_path, "w") as g:
            json.dump(out, g)
