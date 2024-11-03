import json
import os
from abc import ABC
from collections.abc import Callable, Iterable
from typing import cast

import geopandas as gpd
from geopandas import GeoDataFrame

AssetManifest = dict[str, dict[str, list[str]]]


# TODO: Implement a callback system for the various parsing stages.
# Each callback has the same signature which enables it to access all class fields
# and can be composed of various subroutines.
# Example:
# def merge_images(datapath:str, manifest:AssetManifest, geometry:GeoDataFrame) -> None:
#     if subroutine1(datapath):
#       subroutine2(manifest, geometry)
# The class will call each callback sequentially in registration order.


class AssetParser(ABC):
    """Base Asset Parser."""

    def __init__(
        self, dirpath: str, callbacks: Callable | Iterable[Callable] | None = None
    ) -> None:
        """Configure the parser.

        Args:
            dirpath:
                The path to the data directory.
        """
        self._datapath = dirpath
        self._callbacks = [callbacks] if isinstance(callbacks, Callable) else callbacks

        # NOTE: These fields are updated at the beginning of each parsing operation.
        self._manifest: AssetManifest | None
        self._surfaces: GeoDataFrame | None

    @property
    def manifest(self) -> AssetManifest | None:
        return self._manifest

    @property
    def surfaces(self) -> GeoDataFrame | None:
        return self._surfaces

    def parse(self, tile_id: str, overwrite: bool = False) -> None: ...

    # TODO: Refactor this method as a function in a tile utility module.
    def resolve_filepath(self, filename: str) -> str:
        return os.path.join(self._datapath, filename)

    def _update(self, tile_id: str) -> None:
        self._update_manifest(tile_id)
        self._update_surfaces(tile_id)

    def _update_manifest(self, tile_id: str) -> None:
        filepath = self.resolve_filepath(tile_id + ".info.json")
        with open(filepath) as f:
            self._manifest = cast(AssetManifest, json.load(f))

    def _update_surfaces(self, tile_id: str) -> None:
        filepath = self.resolve_filepath(tile_id + ".surf.gpkg")
        self._surfaces = cast(GeoDataFrame, gpd.read_file(filepath))
