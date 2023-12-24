import requests

import config
import utils


# TODO - Reformat, finalize function and variable names, and add documentation.


class DataType:
    ITEM, TILE = range(2)


def download(id_: str, type_: DataType = DataType.TILE) -> None:
    if type_ == DataType.ITEM:
        partial_path = f"{config.env('TEMP_DIR')}{id_}"
        _download_item_data(id_, partial_path)
    elif type_ == DataType.TILE:
        partial_path = f"{config.env('TEMP_DIR')}{id_}"
        _download_tile_data(id_, partial_path)
    else:
        raise ValueError(f"No such data type: {type_}")


def _download_item_data(id_: str, partial_path: str) -> None:
    url = f"{config.var('BAG3D_API_BASE_URL')}{id_}"
    filename = f"{partial_path}{config.var('CITY_JSON')}"

    with requests.Session() as session:
        utils.file.BlockingFileDownloader(
            url, filename, session=session, callbacks=utils.cjio.to_jsonl
        ).download()


def _download_tile_data(id_: str, partial_path: str) -> None:
    # Compile static, compound environment variables.
    # FIXME - Do not recompile this constant at every function call.
    _BASE_TILE_DATA_URL = (
        f"{config.var('BAG3D_TILE_URL')}{config.var('BAG3D_VER')}/tiles/"
    )

    url = f"{_BASE_TILE_DATA_URL}{id_.replace('-', '/')}/{id_}{config.var('CITY_JSON')}"
    filename = f"{partial_path}{config.var('CITY_JSON')}"

    with requests.Session() as session:
        utils.file.BlockingFileDownloader(url, filename, session=session).download()
