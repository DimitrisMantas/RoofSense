from __future__ import annotations

import re
from os import PathLike

import requests
from typing_extensions import override

import config
import utils
from downloaders.base import DataDownloader
from utils.type import BoundingBoxLike


class BAG3DDataDownloader(DataDownloader):
    def __init__(self) -> None:
        super().__init__()

    @override
    def download(self, obj_id: str | BoundingBoxLike) -> None:
        if utils.geom.is_bbox_like(obj_id):
            obj_tp = _ObjectType.BBOX
        else:
            obj_tp = _get_object_type(obj_id)

        tmp_dir = config.env("TEMP_DIR")
        if obj_tp == _ObjectType.BBOX:
            _download_bbox_data(obj_id, base_dir=tmp_dir)
        elif obj_tp == _ObjectType.BUILDING:
            _download_building_data(obj_id, base_dir=tmp_dir)
        else:
            _download_tile_data(obj_id, base_dir=tmp_dir)


class _ObjectType:
    BBOX, BUILDING, TILE = range(3)


def _get_object_type(obj_id: str) -> _ObjectType:
    if re.match(config.var("BUILDING_ID"), obj_id):
        return _ObjectType.BUILDING
    elif re.match(config.var("TILE_ID"), obj_id):
        return _ObjectType.TILE
    raise ValueError(f"Invalid object ID : {obj_id!r}")


# noinspection PyUnusedLocal
def _download_bbox_data(obj_id: BoundingBoxLike, base_dir: str | PathLike) -> None:
    raise NotImplementedError("Only 3DBAG buildings and tiles are currently supported.")


def _download_building_data(obj_id: str, base_dir: str | PathLike) -> None:
    url = f"{config.var('BAG3D_API_BASE_URL')}{obj_id}"
    out_path = f"{base_dir}{obj_id}{config.var('CITY_JSON')}"

    with requests.Session() as s:
        utils.file.BlockingFileDownloader(
            url, out_path, session=s, callbacks=utils.cjio.to_jsonl
        ).download()


def _download_tile_data(obj_id: str, base_dir: str | PathLike) -> None:
    url = (
        f"{config.var('BASE_TILE_ADDRESS')}"
        f"{obj_id.replace('-', '/')}/"
        f"{obj_id}"
        f"{config.var('CITY_JSON')}"
    )
    out_path = f"{base_dir}{obj_id}{config.var('CITY_JSON')}"

    with requests.Session() as s:
        utils.file.BlockingFileDownloader(url, out_path, session=s).download()
