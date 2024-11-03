import os

import dotenv
import geopandas as gpd

from roofsense.utils.file import mkdirs

# TODO - Reformat, finalize function and variable names, and add documentation.

_PROJ_DIR = os.path.abspath(os.path.dirname(__file__))


def config(training: bool = False) -> None:
    dotenv.load_dotenv(os.path.join(_PROJ_DIR, "config.env"))

    # Enable GeoPandas speed-ups.
    gpd.options.io_engine = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"

    mkdirs(env("TEMP_DIR"))
    mkdirs(env("LOG_DIR"))
    if training:
        mkdirs(f"{env('ORIGINAL_DATA_DIR')}{var('TRAINING_IMAG_DIRNAME')}")
        mkdirs(f"{env('ORIGINAL_DATA_DIR')}{var('TRAINING_CHIP_DIRNAME')}")


def env(key: str) -> str:
    return os.path.join(_PROJ_DIR, os.environ[key])


def var(key: str) -> str:
    return os.environ[key]


def default_data_dict() -> dict:
    return {var("DEFAULT_ID_FIELD_NAME"): [], var("DEFAULT_GM_FIELD_NAME"): []}


def default_data_tabl(data: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(data, crs=var("CRS"))
