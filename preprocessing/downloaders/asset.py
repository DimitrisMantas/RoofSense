from __future__ import annotations

import json
import os.path
from os import PathLike
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import requests
from typing_extensions import override

import config
import utils
from preprocessing.downloaders._base import _Downloader


class AssetDownloader(_Downloader):
    """Convenience class for downloading 3DBAG tile assets (i.e., from the
    latest BM5, 8 cm RGB orthoimagery and AHN point cloud collections)."""

    def __init__(self) -> None:
        """Initialize the downloader.

        .. note::
            This operation requires that the BM5 and AHN tile indices are present at
            the locations specified by the ``IMAGE_TILE_INDEX`` and
            ``LIDAR_TILE_INDEX`` configuration parameters, respectively.
        """
        self._image_info_store = _InfoStore(config.env("IMAGE_SHEET_INDEX"))
        self._lidar_info_store = _InfoStore(config.env("LIDAR_SHEET_INDEX"))

    # TODO: Document this method.
    @override
    def download(self, tile_id: str) -> None:
        image_info = self._image_info_store.get_info(tile_id)
        lidar_info = self._lidar_info_store.get_info(tile_id)

        manifest = self._gen_manifest(tile_id, image_info, lidar_info)

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
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
        ],
        lidar_info: tuple[
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
        ],
    ) -> Optional[dict[str, dict[str, list[str]]]]:
        # TODO: Think of a better way to compile paths from individual user-defined
        #  parameters.
        dst_filepath = os.path.join(config.env("TEMP_DIR"), tile_id + ".info.json")

        if os.path.isfile(dst_filepath):
            with open(dst_filepath) as src:
                manifest = json.load(src)
        else:
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
    def __init__(self, index_path: str | bytes | PathLike) -> None:
        self._index: gpd.GeoDataFrame = gpd.read_file(index_path)

    def get_info(
        self, tile_id: str
    ) -> tuple[
        np.ndarray[tuple[Any,], np.dtype[np.object_]],
        np.ndarray[tuple[Any,], np.dtype[np.object_]],
    ]:
        # NOTE: This operation implicitly requires that the roof surfaces have
        # already been parsed.
        surfs = utils.geom.read_surfaces(tile_id)

        hits = self._index.overlay(surfs)
        # NOTE: The matches initially contain duplicate values because each surface
        # is individually intersected with the index.
        # This approach is actually faster than first dissolving the surfaces.
        return hits.tid.unique(), hits.url.unique()

    @staticmethod
    def gen_manifest(
        info: tuple[
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
        ],
    ) -> dict[str, list[str]]:
        manifest = {
            "tid": info[0].tolist(),
            "url": info[1].tolist(),
        }
        return manifest
