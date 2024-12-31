import os
from abc import ABC
from collections.abc import Callable, Iterable

from geopandas import GeoDataFrame

from roofsense.bag3d import BAG3DTileStore, LevelOfDetail, TileAssetManifest


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
        self,
        store: BAG3DTileStore,
        callbacks: Callable | Iterable[Callable] | None = None,
    ) -> None:
        """Configure the parser.

        Args:
            dirpath:
                The path to the data directory.
        """
        self.store: BAG3DTileStore = store
        self._callbacks = [callbacks] if isinstance(callbacks, Callable) else callbacks

        # NOTE: These fields are updated at the beginning of each parsing operation.
        self._manifest: TileAssetManifest | None
        self._surfaces: GeoDataFrame | None

    @property
    def manifest(self) -> TileAssetManifest:
        return self._manifest

    @property
    def surfaces(self) -> GeoDataFrame | None:
        return self._surfaces

    def parse(self, tile_id: str, overwrite: bool = False) -> None: ...

    # TODO: Refactor this method as a function in a tile utility module.
    def resolve_filepath(self, filename: str) -> str:
        return os.path.join(self.store.dirpath, filename)

    def _update(self, tile_id: str) -> None:
        self._update_manifest(tile_id)
        self._update_surfaces(tile_id)

    def _update_manifest(self, tile_id: str) -> None:
        self._manifest = self.store.asset_manifest(tile_id)

    def _update_surfaces(self, tile_id: str) -> None:
        self._surfaces = self.store.read_tile(tile_id, lod=LevelOfDetail.LoD22)
