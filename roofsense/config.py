import os

import dotenv
import geopandas as gpd

_PROJ_DIR = os.path.abspath(os.path.dirname(__file__))


def config() -> None:
    dotenv.load_dotenv(os.path.join(_PROJ_DIR, "config.env"))

    # Enable GeoPandas speed-ups.
    gpd.options.io_engine = "pyogrio"
    os.environ.update({"PYOGRIO_USE_ARROW": "1"})

    # See https://github.com/pangeo-data/cog-best-practices for more information.
    os.environ.update(
        {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "AWS_NO_SIGN_REQUEST": "YES",
            "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
            "GDAL_SWATH_SIZE": "200000000",
            "VSI_CURL_CACHE_SIZE": "200000000",
        }
    )


def env(key: str) -> str:
    return os.path.join(_PROJ_DIR, os.environ[key])
