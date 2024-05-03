from __future__ import annotations

import json
import pathlib
from abc import ABC, abstractmethod

import geopandas as gpd

import config
import utils


class DataParser(ABC):
    @abstractmethod
    def parse(self, obj_id: str) -> None: ...


class AssetParser(DataParser):
    def __init__(self) -> None:
        super().__init__()
        self._manifest: dict[str, dict[str, list[str]]] | None = None
        self._surfs: gpd.GeoDataFrame | None = None

    @abstractmethod
    def parse(self, obj_id: str) -> None: ...

    def _update(self, obj_id: str) -> None:
        manifest_path = (
            f"{config.var('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('ASSET_MANIFEST_EXTENSION')}"
            f"{config.var('JSON')}"
        )
        with pathlib.Path(manifest_path).open() as f:
            self._manifest = json.load(f)
        self._surfs = utils.geom.read_surfaces(obj_id)
