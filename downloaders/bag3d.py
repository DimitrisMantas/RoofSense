from __future__ import annotations

import re

import requests
from typing_extensions import override

import config
import utils
from downloaders.base import DataDownloader
from utils.type import BoundingBoxLike


class BAG3DDataDownloader(DataDownloader):
    def __init__(self) -> None:
        # TODO: Check whether having a state-less class only to maintain a consistent
        #       API is a design smell.
        super().__init__()

    @override
    def download(self, obj_id: str | BoundingBoxLike) -> None:
        if utils.geom.is_bbox_like(obj_id):
            obj_tp = _ObjectType.BBOX
        else:
            obj_tp = _find_object_type(obj_id)

        base_dir = config.env("TEMP_DIR")
        if obj_tp == _ObjectType.BBOX:
            _download_bbox_data(obj_id, base_dir)
        elif obj_tp == _ObjectType.BUILDING:
            _download_building_data(obj_id, base_dir)
        else:
            _download_tile_data(obj_id, base_dir)


class _ObjectType:
    BBOX, BUILDING, TILE = range(3)


def _find_object_type(obj_id: str) -> _ObjectType:
    if re.match(config.var("BUILDING_ID"), obj_id):
        return _ObjectType.BUILDING
    elif re.match(config.var("TILE_ID"), obj_id):
        return _ObjectType.TILE
    raise ValueError(f"Invalid object ID : {obj_id}")


def _download_bbox_data(obj_id: BoundingBoxLike, dirname: str) -> None:
    raise NotImplementedError


def _download_tile_data(obj_id: str, base_dir: str) -> None:
    addr = (
        f"{config.var('BASE_TILE_ADDRESS')}"
        f"{obj_id.replace('-', '/')}/"
        f"{obj_id}"
        f"{config.var('CITY_JSON')}"
    )
    path = f"{base_dir}{obj_id}{config.var('CITY_JSON')}"

    with requests.Session() as s:
        utils.file.BlockingFileDownloader(addr, path, session=s).download()


def _download_building_data(obj_id: str, base_dir: str) -> None:
    addr = f"{config.var('BAG3D_API_BASE_URL')}{obj_id}"
    path = f"{base_dir}{obj_id}{config.var('CITY_JSON')}"

    with requests.Session() as s:
        utils.file.BlockingFileDownloader(
            addr, path, session=s, callbacks=utils.cjio.to_jsonl
        ).download()
