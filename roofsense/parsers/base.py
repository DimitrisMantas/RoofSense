from __future__ import annotations

import os
from collections.abc import Callable, Iterable

from geopandas import GeoDataFrame

from roofsense.bag3d import BAG3DTileStore, LevelOfDetail, TileAssetManifest


class BAG3DTileAssetParser:
    """Parser for generic 3DBAG tile assets."""

    def __init__(
        self,
        tile_store: BAG3DTileStore,
        callbacks: BAG3DTileAssetParsingStage
        | Iterable[BAG3DTileAssetParsingStage]
        | None,
    ) -> None:
        """Initialize the importer.

        Args:
            tile_store:
                The BAG3D tile store used to import the dataset.
            callbacks:
                One or more callback functions representing different parsing stages.
                If multiple callbacks are provided, they are executed sequentially in the order in which they are registered.
                Each callback must accept an 'AssetParser' instance as its first argument, a string representing a 3DBAG tile ID as its second, and a boolean representing whether to overwrite potential intermediate outputs as its third and final argument, respectively.
                If other arguments are necessary, they can be squashed using 'functools.partial' or the parser can be appropriately extended (e.g., to include the required arguments as class or instance attributes).
                The callback must return 'None'.
                If an intermediate output is necessary (e.g., to be processed by a later callback), it should be serialized and written to disk.
                All callbacks must have the same signature.
                A parser with no registered callbacks does not perform any parsing.
        """
        self._tile_store = tile_store
        if isinstance(callbacks, Iterable) or callbacks is None:
            self._callbacks = callbacks
        elif callable(callbacks):
            self._callbacks = [callbacks]
        else:
            msg = "callbacks must be callable or an iterable of callables or none."
            raise ValueError(msg)

        # NOTE: These fields are updated at the beginning of each parsing operation.
        self._manifest: TileAssetManifest | None
        self._surfaces: GeoDataFrame | None

    @property
    def tile_store(self) -> BAG3DTileStore:
        return self._tile_store

    @property
    def manifest(self) -> TileAssetManifest:
        return self._manifest

    @property
    def surfaces(self) -> GeoDataFrame | None:
        return self._surfaces

    def parse(self, tile_id: str, overwrite: bool = False) -> None:
        self._update(tile_id)
        self._execute(tile_id, overwrite)

    def register_callbacks(
        self,
        callback: BAG3DTileAssetParsingStage,
        *callbacks: Iterable[BAG3DTileAssetParsingStage],
    ) -> None:
        self._callbacks.extend([callback] + list(callbacks))

    # TODO: Check that removing callbacks by name works.
    def remove_callbacks(
        self,
        callback: int | BAG3DTileAssetParsingStage,
        *callbacks: Iterable[int | BAG3DTileAssetParsingStage],
    ) -> None:
        to_remove = []
        for callback in [callback] + list(callbacks):
            if isinstance(callback, int):
                to_remove.append(callback)
            elif callable(callback):
                to_remove.append(self._callbacks.index(callback))
        for i in to_remove:
            self._callbacks.pop(i)

    def resolve_filepath(self, filename: str) -> str:
        return os.path.join(self._tile_store.dirpath, filename)

    def _execute(self, tile_id: str, overwrite: bool = False) -> None:
        if self._callbacks is None:
            return
        for callback in self._callbacks:
            callback(self, tile_id, overwrite)

    def _update(self, tile_id: str) -> None:
        self._update_manifest(tile_id)
        self._update_surfaces(tile_id)

    # TODO: Should the update routines be their own methods?
    def _update_manifest(self, tile_id: str) -> None:
        self._manifest = self._tile_store.asset_manifest(tile_id)

    def _update_surfaces(self, tile_id: str) -> None:
        self._surfaces = self._tile_store.read_tile(tile_id, lod=LevelOfDetail.LoD22)


BAG3DTileAssetParsingStage = Callable[[BAG3DTileAssetParser, str, bool], None]
