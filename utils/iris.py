from __future__ import annotations

import copy
import json
import pathlib
from os import PathLike
from typing import Optional

import rasterio

import config
from utils.type import JSONLike


class ConfigurationFile:
    def __init__(self) -> None:
        self._base = read_base()

    def create(self, filename: str | PathLike[str]) -> None:
        # TODO: Consider splitting the components of this method into separate ones.
        p = pathlib.Path(filename)

        f: rasterio.io.DatasetReader
        with rasterio.open(p) as f:
            # NOTE: IRIS expects image shapes in reverse NumPy order.
            shape = [f.width, f.height]

        j = copy.deepcopy(self._base)
        j["images"]["shape"] = shape

        with open(f"{p.stem}{config.var('JSON')}", "w") as g:
            json.dump(j, g)


def read_base() -> Optional[JSONLike]:
    with open(config.var("IRIS_BASE_CONFIG_FILENAME")) as f:
        return json.load(f)
