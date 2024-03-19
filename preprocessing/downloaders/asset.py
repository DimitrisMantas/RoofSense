from __future__ import annotations

import json
import pathlib
from os import PathLike
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import requests

import config
import utils


class AssetDownloader:
    def __init__(self) -> None:
        self._image_info_store = _InfoStore(config.env("IMAGE_SHEET_INDEX"))
        self._lidar_info_store = _InfoStore(config.env("LIDAR_SHEET_INDEX"))

    def download(self, tile_id: str) -> None:
        image_info = self._image_info_store.get_info(tile_id)
        lidar_info = self._lidar_info_store.get_info(tile_id)

        manifest = self._gen_manifest(tile_id, image_info, lidar_info)

        urls = manifest["image"]["urls"] + manifest["lidar"]["urls"]
        dst_filepaths = [
            pathlib.Path(config.env("TEMP_DIR")) / pathlib.Path(url).name
            for url in urls
        ]

        with requests.Session() as session:
            utils.file.ThreadedFileDownloader(
                urls, filenames=dst_filepaths, session=session
            ).download()

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
        dst_filepath = pathlib.Path(
            config.env("TEMP_DIR")
        ) / f'{tile_id}{config.var("ASSET_MANIFEST_EXTENSION")}{config.var("JSON")}'

        if dst_filepath.is_file():
            with dst_filepath.open() as src:
                return json.load(src)

        manifest = {
            "image": self._image_info_store.gen_manifest(image_info),
            "lidar": self._lidar_info_store.gen_manifest(lidar_info),
        }

        with dst_filepath.open("w") as dst:
            json.dump(manifest, dst)

        return manifest


# TODO: Check whether this is an appropriate name given the purpose of this class.
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
        # is separately intersected with the index.
        return hits.tid.unique(), hits.url.unique()

    @staticmethod
    def gen_manifest(
        info: tuple[
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
            np.ndarray[tuple[Any,], np.dtype[np.object_]],
        ],
    ) -> dict[str, list[str]]:
        manifest = {
            "tids": info[0].tolist(),
            "urls": info[1].tolist(),
        }

        return manifest


if __name__ == "__main__":
    config.config()

    AssetDownloader().download("9-284-556")
