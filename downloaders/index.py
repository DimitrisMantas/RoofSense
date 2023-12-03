import os
import re
import typing

import geopandas as gpd
import requests
import shapely

import config
import utils

# TODO - Reformat, finalize function and variable names, and add documentation.


# NOTE - This pattern detects alphanumeric characters and the forward slash.
ID_FMT = re.compile(r"[^\d/]+", re.IGNORECASE)


class TileIndex(gpd.GeoDataFrame):
    def __init__(self, overwrite: typing.Optional[bool] = False) -> None:
        if utils.file.exists(config.env("BAG3D_INDEX_FILENAME")) and not overwrite:
            super().__init__(gpd.read_file(config.env("BAG3D_INDEX_FILENAME")))
        else:
            super().__init__(_reconstruct(), crs=os.environ["DEFAULT_CRS"])


# TOSELF: The download operation involves fetching and returning web data as-is in this
#         module but fetching *and* writing to disk in another.
#
# TODO: Come up with a common terminology.
def _download() -> utils.type.BAG3DTileIndexJSON:
    # TODO: Include this variable in the runtime configuration file.
    path = f"{os.environ['BAG3D_WFS_BASE_URL']}&version={os.environ['BAG3D_WFS_VERSION']}&request={os.environ['BAG3D_WFS_REQUEST']}&typeNames={os.environ['BAG3D_WFS_TYPENAMES']}&outputFormat={os.environ['BAG3D_WFS_OUTPUT_FORMAT']}"

    # NOTE: This download should be blocking because the 3DBAG index is required to
    #       operate in tile mode.
    # TOSELF: Use the BlockingDownloader and parse the response inside a hook?
    response = requests.get(path)
    response.raise_for_status()
    return response.json()


def _parse_tile_id(tile: utils.type.BAG3TileData) -> str:
    return ID_FMT.sub("", tile[os.environ["DEFAULT_ID_FIELD_NAME"]]).replace("/", "-")


def _parse_tile_gm(tile: utils.type.BAG3TileData) -> shapely.Polygon:
    return shapely.Polygon(
        tile[os.environ["DEFAULT_GM_FIELD_NAME"]][os.environ["BAG3D_TILE_COORDINATES"]][
            0])


def _pase_tile(tile: utils.type.BAG3TileData,
               data: utils.type.BAG3DTileIndexData) -> None:
    data[os.environ["DEFAULT_ID_FIELD_NAME"]].append(_parse_tile_id(tile))
    data[os.environ["DEFAULT_GM_FIELD_NAME"]].append(_parse_tile_gm(tile))


def _parse(js: utils.type.BAG3DTileIndexJSON) -> utils.type.BAG3DTileIndexData:
    data = config.default_data_dict()
    for tile in js[os.environ["BAG3D_TILE_FEATURES"]]:
        _pase_tile(tile, data)
    return data


def _reconstruct() -> utils.type.BAG3DTileIndexData:
    return _parse(_download())
