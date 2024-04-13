from __future__ import annotations

import json
import os
from typing import Any

import geopandas as gpd
import numpy as np
import requests
from typing_extensions import override

import config
import utils
from common.downloaders._base import _Downloader


class AssetDownloader(_Downloader):
    """Convenience class for downloading 3DBAG tile assets from the latest BM5,
    8 cm RGB orthoimagery, and AHN point cloud collections."""

    def __init__(self) -> None:
        """Initialize the downloader.

        .. note::
            This operation requires that the BM5 and AHN tile indices are present at
            the locations specified by the ``IMAGE_TILE_INDEX`` and
            ``LIDAR_TILE_INDEX`` configuration parameters, respectively.
        """
        self._image_info_store = _InfoStore(config.env("IMAGE_SHEET_INDEX"))
        self._lidar_info_store = _InfoStore(config.env("LIDAR_SHEET_INDEX"))

    # TODO: Document the asset manifest.
    @override
    def download(self, tile_id: str) -> None:
        """Download the assets of a single 3DBAG tile.

        :param tile_id: The tile ID (e.g., ``"9-284-556"``).

        .. note::
            BM5 and AHN tiles are downloaded in the `GeoTIFF
            <https://en.wikipedia.org/wiki/GeoTIFF>`_ and `LASzip
            <https://rapidlasso.de/laszip/>`_ file format, respectively.

            Tile assets are saved in the directory specified by the ``TEMP_DIR``
            configuration parameter, and named using their ID (e.g.,
            ``2023_084000_446000_RGB_hrl.tif`` and ``37EN1_15.LAZ`` for BM5 and AHN4
            tiles, respectively).
        """
        # Generate the asset manifest.
        manifest = self._gen_manifest(
            tile_id,
            self._image_info_store.get_info(tile_id),
            self._lidar_info_store.get_info(tile_id),
        )

        # Download the assets.
        urls = manifest["image"]["url"] + manifest["lidar"]["url"]
        dst_filepaths = [
            os.path.join(config.env("TEMP_DIR"), os.path.basename(url)) for url in urls
        ]

        with requests.Session() as session:
            utils.file.ThreadedFileDownloader(
                urls, filenames=dst_filepaths, session=session
            ).download()

    # TODO: Document this method.
    def _gen_manifest(
        self,
        tile_id: str,
        image_info: tuple[
            np.ndarray[tuple[Any], np.dtype[np.object_]],
            np.ndarray[tuple[Any], np.dtype[np.object_]],
        ],
        lidar_info: tuple[
            np.ndarray[tuple[Any], np.dtype[np.object_]],
            np.ndarray[tuple[Any], np.dtype[np.object_]],
        ],
    ) -> dict[str, dict[str, list[str]]] | None:
        # TODO: Think of a better way to compile paths from individual user-defined
        #  parameters.
        dst_filepath = os.path.join(config.env("TEMP_DIR"), tile_id + ".info.json")

        if os.path.isfile(dst_filepath):
            # Load the manifest.
            with open(dst_filepath) as src:
                manifest = json.load(src)
        else:
            # Generate the manifest.
            manifest = {
                "image": self._image_info_store.gen_manifest(image_info),
                "lidar": self._lidar_info_store.gen_manifest(lidar_info),
            }

            with open(dst_filepath, mode="w") as dst:
                json.dump(manifest, dst)

        return manifest


# TODO: Check whether this is an appropriate name given the purpose of this class.
# TODO: Document this class.
class _InfoStore:
    def __init__(self, index_path: str | bytes | os.PathLike) -> None:
        self._index: gpd.GeoDataFrame = gpd.read_file(index_path)

    def get_info(
        self, tile_id: str
    ) -> tuple[
        np.ndarray[tuple[Any], np.dtype[np.object_]],
        np.ndarray[tuple[Any], np.dtype[np.object_]],
    ]:
        # TODO: Move this note to the ``AssetDownloader.download()`` docstring.
        # NOTE: This operation implicitly requires that the roof surfaces have
        #       already been parsed.
        surfs = utils.geom.read_surfaces(tile_id)

        hits = self._index.overlay(surfs)
        # NOTE: The result of the index-surface intersection operation initially
        #       includes duplicate tiles
        #       because each surface is intersected with the index individually.
        #       However, this approach is actually faster than dissolving the surfaces
        #       first.
        return hits.tid.unique(), hits.url.unique()

    @staticmethod
    def gen_manifest(
        info: tuple[
            np.ndarray[tuple[Any], np.dtype[np.object_]],
            np.ndarray[tuple[Any], np.dtype[np.object_]],
        ],
    ) -> dict[str, list[str]]:
        return {
            "tid": info[0].tolist(),
            "url": info[1].tolist(),
        }
