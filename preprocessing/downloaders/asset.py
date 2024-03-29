from __future__ import annotations

import json
import pathlib
import urllib.parse

import geopandas as gpd
import requests
from typing_extensions import override

import config
import utils
from preprocessing.downloaders.base import DataDownloader
from utils.type import BoundingBoxLike


class AssetDownloader(DataDownloader):
    def __init__(self) -> None:
        super().__init__()
        self._index = gpd.read_file(config.env("ASSET_SHEET_INDEX"))

    @override
    def download(self, obj_id: str | BoundingBoxLike) -> None:
        img_ids, ldr_ids = self._get_asset_ids(obj_id)
        _write_asset_manifest(obj_id, img_ids, ldr_ids)
        # TODO: Parallelize this operation.
        _download_image_assets(img_ids)
        _download_lidar_assets(ldr_ids)

    def _get_asset_ids(
        self, obj_id: str | BoundingBoxLike
    ) -> tuple[list[str], list[str]]:
        surfs = utils.geom.read_surfaces(obj_id)
        ids = self._index.overlay(surfs)
        return (
            ids[config.var("ASSET_INDEX_IMAGE_IDS")].unique().tolist(),
            ids[config.var("ASSET_INDEX_LIDAR_IDS")].unique().tolist(),
        )


def _write_asset_manifest(obj_id: str, img_ids: list[str], ldr_ids: list[str]) -> None:
    # TODO: Create this path using a utility function ro avoid code duplication.
    out_path = (
        f"{config.var('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('ASSET_MANIFEST_EXTENSION')}"
        f"{config.var('JSON')}"
    )
    if utils.file.exists(out_path):
        return
    manifest = {
        config.var("ASSET_MANIFEST_IMAGE_IDS"): [
            f"{img_id}{config.var('TIFF')}" for img_id in img_ids
        ],
        config.var("ASSET_MANIFEST_LIDAR_IDS"): [
            f"{ldr_id}{config.var('LAZ')}" for ldr_id in ldr_ids
        ],
    }
    with pathlib.Path(out_path).open("w") as f:
        json.dump(manifest, f)


def _download_image_assets(ids: list[str]) -> None:
    urls = [
        f"{config.var('BASE_IMAGE_DATA_URL')}{img_tp}_{img_id}{config.var('TIFF')}"
        for img_tp in [config.var("CIR_IDENTIFIER"), config.var("RGB_IDENTIFIER")]
        for img_id in ids
    ]
    out_paths = _get_output_paths(urls)
    with requests.Session() as s:
        utils.file.ThreadedFileDownloader(urls, out_paths, session=s).download()


def _download_lidar_assets(ids: list[str]) -> None:
    urls = [
        f"{config.var('BASE_LIDAR_DATA_URL')}{ldr_id}{config.var('LAZ')}"
        for ldr_id in ids
    ]
    out_paths = _get_output_paths(urls)
    with requests.Session() as s:
        utils.file.ThreadedFileDownloader(urls, out_paths, session=s).download()


# TODO: Move this function to a more appropriate module.
def _get_output_paths(urls: list[str]) -> list[str]:
    return [
        f"{config.var('TEMP_DIR')}{urllib.parse.urlparse(url).path.rsplit('/')[-1]}"
        for url in urls
    ]
