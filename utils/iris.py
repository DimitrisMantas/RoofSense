from __future__ import annotations

import copy
import json

import rasterio

import config
from utils.type import JSONLike


class ConfigurationFile:
    def __init__(self) -> None:
        self._base = _read_base()

    def create(self, obj_id: str) -> None:
        img_path = f"{config.env('TEMP_DIR')}{obj_id}.stack{'.tif'}"
        out_path = f"{config.env('TEMP_DIR')}{obj_id}.iris{'.json'}"

        f: rasterio.io.DatasetReader
        with rasterio.open(img_path) as f:
            # NOTE: IRIS expects image shapes in reverse NumPy order.
            shape = [f.width, f.height]

        j = copy.deepcopy(self._base)
        j["images"]["shape"] = shape
        j["segmentation"]["mask_area"] = [0, 0, *shape]

        with open(out_path, "w") as g:
            json.dump(j, g)


def _read_base() -> JSONLike:
    with open(config.var("IRIS_BASE_CONFIG_FILENAME")) as f:
        return json.load(f)
