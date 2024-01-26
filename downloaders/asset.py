import json
import pathlib
import urllib.parse

import geopandas as gpd
import requests
from typing_extensions import override

import config
import utils
from downloaders.base import DataDownloader


class AssetDataDownloader(DataDownloader):
    def __init__(self) -> None:
        super().__init__(config.env("ASSET_SHEET_INDEX"))

    @override
    def download(self, obj_id: str) -> None:
        # TODO: See if there is a significant difference in performance between
        #       intersecting the index with individual buffers versus their union.
        surfs = utils.geom.buffer(utils.geom.read_surfaces(obj_id))

        img_ids, ldr_ids = self._find_asset_ids(surfs)

        _write_asset_manifest(obj_id, img_ids, ldr_ids)
        # TODO: Parallelize this operation.
        _download_image_data(img_ids)
        _download_lidar_data(ldr_ids)

    def _find_asset_ids(self, geom: gpd.GeoDataFrame) -> tuple[list[str], list[str]]:
        matches = self._index.overlay(geom)
        return (
            matches["image_id"].unique().tolist(),
            matches["lidar_id"].unique().tolist(),
        )


def _write_asset_manifest(
    obj_id: str, image_ids: list[str], lidar_ids: list[str]
) -> None:
    path = (
        f"{config.var('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('ASSET_MANIFEST_EXTENSION')}"
        f"{config.var('JSON')}"
    )
    if utils.file.exists(path):
        return

    # TODO: Read the manifest sceleton from a relevant environment variable.
    manifest = {"image_ids": image_ids, "lidar_ids": lidar_ids}
    with pathlib.Path(path).open("w") as f:
        json.dump(manifest, f)


def _download_image_data(ids: list[str]) -> None:
    addrs = [
        f"{config.var('BASE_IMAGE_DATA_URL')}{img_tp}_{img_id}{config.var('TIFF')}"
        # TODO: Read the image types from a relevant environment variable.
        for img_tp in ["RGB", "CIR"]
        for img_id in ids
    ]
    paths = _build_paths(addrs)

    with requests.Session() as s:
        utils.file.ThreadedFileDownloader(addrs, paths, session=s).download()


def _download_lidar_data(ids: list[str]) -> None:
    addrs = [
        f"{config.var('BASE_LIDAR_DATA_URL')}{ldr_id}{config.var('LAZ')}"
        for ldr_id in ids
    ]
    paths = _build_paths(addrs)

    with requests.Session() as s:
        utils.file.ThreadedFileDownloader(addrs, paths, session=s).download()


# TODO: Consider moving this function to a more appropriate location.
def _build_paths(addrs: list[str]) -> list[str]:
    return [
        f"{config.var('TEMP_DIR')}{urllib.parse.urlparse(addr).path.rsplit('/')[-1]}"
        for addr in addrs
    ]
