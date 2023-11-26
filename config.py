import os
import typing

import dotenv
import geopandas as gpd

import utils

# TODO - Reformat, finalize function and variable names, and add documentation.

_PROJ_DIR = os.path.abspath(os.path.dirname(__file__))


def config() -> None:
    dotenv.load_dotenv(os.path.join(_PROJ_DIR, "config.env"))

    # Enable GeoPandas speed-ups.
    gpd.options.io_engine = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"

    utils.file.mkdirs(env("AHN34_INDEX_DIR"))
    utils.file.mkdirs(env("BAG3D_INDEX_DIR"))
    utils.file.mkdirs(env("ORTHO_INDEX_DIR"))

    utils.file.mkdirs(env("TEMP_DIR"))
    utils.file.mkdirs(env("LOG_DIR"))


def env(key: str) -> typing.Optional[str]:
    return _PROJ_DIR + "/" + os.environ[key]


def var(key: str) -> typing.Optional[str]:
    return os.environ[key]


def default_data_dict() -> dict:
    return {var("DEFAULT_ID_FIELD_NAME"): [], var("DEFAULT_GM_FIELD_NAME"): []}


def default_data_tabl(data: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(data, crs=var("DEFAULT_CRS"))
