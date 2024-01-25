from __future__ import annotations

import re

import requests

import config
from downloaders._base import DataDownloader
from utils.cjio import to_jsonl
from utils.file import BlockingFileDownloader
from utils.type import BoundingBoxLike


class BAG3DDataDownloader(DataDownloader):
    def __init__(self) -> None:
        # TODO: Figure out whether this class should exist or not.
        super().__init__()

    # TODO: Find out why this decorator can be imported from the typing module.
    def download(self, obj_id: str | BoundingBoxLike) -> None:
        if _is_bbox_like(obj_id):
            obj_type = _ObjectType.BBOX
        else:
            obj_type = _get_object_type(obj_id)

        dirname = f"{config.env('TEMP_DIR')}"
        if obj_type == _ObjectType.BBOX:
            _download_bbox_data(obj_id, dirname)
        elif obj_type == _ObjectType.ITEM:
            _download_item_data(obj_id, dirname)
        else:
            _download_tile_data(obj_id, dirname)


# TODO: Promote these static constants to environment variables.
_ITEM_ID = r"^NL\.IMBAG\.Pand\.\d{16}$"
_TILE_ID = r"^\d{1,3}-\d{1,3}-\d{1,3}$"


class _ObjectType:
    BBOX, ITEM, TILE = range(3)


def _is_bbox_like(obj_id: str):
    return (
        isinstance(obj_id, BoundingBoxLike.__origin__)
        and len(obj_id) == 4
        and all([(isinstance(i, int) or isinstance(i, float)) for i in BoundingBoxLike])
    )


def _get_object_type(obj_id: str) -> _ObjectType:
    if re.match(_ITEM_ID, obj_id):
        return _ObjectType.ITEM
    elif re.match(_TILE_ID, obj_id):
        return _ObjectType.TILE
    raise ValueError(f"Invalid object ID : {obj_id}")


def _download_bbox_data(obj_id: BoundingBoxLike, dirname: str) -> None:
    raise NotImplementedError


def _download_item_data(obj_id: str, dirname: str) -> None:
    # TODO: Harmonize these variable names.
    request = f"{config.var('BAG3D_API_BASE_URL')}{obj_id}"
    pathnam = f"{dirname}{obj_id}{config.var('CITY_JSON')}"

    with requests.Session() as session:
        BlockingFileDownloader(
            request, pathnam, session=session, callbacks=to_jsonl
        ).download()


def _download_tile_data(obj_id: str, dirname: str) -> None:
    # TODO: Promote this static constant to an environment variable.
    _BASE_TILE_ADDRESS = (
        f"{config.var('BAG3D_TILE_URL')}{config.var('BAG3D_VER')}/tiles/"
    )

    # TODO: Harmonize these variable names.
    request = (
        f"{_BASE_TILE_ADDRESS}"
        f"{obj_id.replace('-', '/')}/"
        f"{obj_id}"
        f"{config.var('CITY_JSON')}"
    )
    pathnam = f"{dirname}{obj_id}{config.var('CITY_JSON')}"

    with requests.Session() as session:
        BlockingFileDownloader(request, pathnam, session=session).download()
