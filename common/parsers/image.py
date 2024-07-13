import json
import os

import geopandas as gpd
import numpy as np
import rasterio.merge
from geopandas import GeoDataFrame

from utils import raster
from utils.file import confirm_write_op

AssetManifest = dict[str, dict[str, list[str]]]


# TODO: Implement a callback system for the various parsing stages.
# Each callback has the same signature which enables it to access all class fields
# and can be composed of various subroutines.
# Example:
# def merge_images(datapath:str, manifest:AssetManifest, geometry:GeoDataFrame) -> None:
#     if subroutine1(datapath):
#       subroutine2(manifest, geometry)
# The class will call each callback sequentially in registration order.


class ImageParser:
    """BM5 Tile Parser."""

    def __init__(self, datapath: str) -> None:
        """Configure the parser.

        Args:
            datapath:
                The path to the data directory.
        """
        self._datapath = datapath

        # NOTE: These fields are updated at the beginning of each parsing operation.
        self._manifest: AssetManifest | None
        self._geometry: GeoDataFrame | None

    def parse(self, tile_id: str, overwrite: bool = False) -> None:
        """Parse the BM5 tiles corresponding to a particular 3DBAG tile.

        This method merges the input rasters to the tile bounds.using a reverse
        painter's algorithm.
        The local coordinates of the output pixel centers are guaranteed to be integer
        multiples of the respective GSD such that the constituent rasters are aligned.
        This means that the resulting image bounds are at least equal to those of
        the underlying geometry.
        See ``rasterio.merge.merge`` for more implementation details.

        Warnings:
            This method requires that the provided tile has already been parsed and
            that the corresponding asset manifest and roof surface geometry exist in
            the specified data directory.

        Args:
            tile_id:
                The tile ID.
            overwrite:
                Whether to overwrite any previous output.
        """
        self._update(tile_id)

        # TODO: Consider refactoring this block into a separate method.
        dst_path = self._resolve_filepath(tile_id + ".rgb.tif")
        if not confirm_write_op(dst_path, type="file", overwrite=overwrite):
            return

        src_paths = [
            self._resolve_filepath(id_ + ".tif")
            for id_ in self._manifest["image"]["tid"]
        ]
        rasterio.merge.merge(
            src_paths,
            bounds=self._geometry.total_bounds.tolist(),
            target_aligned_pixels=True,
            dst_path=dst_path,
            dst_kwds=raster.DefaultProfile(
                # NOTE: The data type is specified in order for a descriptor to be
                # able to be assigned to the output profile.
                dtype=np.uint8
            ),
        )

    # TODO: Refactor this method as a function in a tile utility module.
    def _resolve_filepath(self, filename: str) -> str:
        return os.path.join(self._datapath, filename)

    def _update(self, tile_id: str) -> None:
        self._update_manifest(tile_id)
        self._update_geometry(tile_id)

    def _update_manifest(self, tile_id: str) -> None:
        filepath = self._resolve_filepath(tile_id + ".info.json")
        with open(filepath) as f:
            self._manifest = json.load(f)

    def _update_geometry(self, tile_id: str) -> None:
        filepath = self._resolve_filepath(tile_id + ".surf.gpkg")
        self._geometry = gpd.read_file(filepath)
