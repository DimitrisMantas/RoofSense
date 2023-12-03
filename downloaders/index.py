#          Copyright © 2023 Dimitris Mantas
#
#          This file is part of RoofSense.
#
#          This program is free software: you can redistribute it and/or modify
#          it under the terms of the GNU General Public License as published by
#          the Free Software Foundation, either version 3 of the License, or
#          (at your option) any later version.
#
#          This program is distributed in the hope that it will be useful,
#          but WITHOUT ANY WARRANTY; without even the implied warranty of
#          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#          GNU General Public License for more details.
#
#          You should have received a copy of the GNU General Public License
#          along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
_TILE_ID_FMT = re.compile(r"[^\d/]+", re.IGNORECASE)


class TileIndex(gpd.GeoDataFrame):
    def __init__(self, reconstruct: typing.Optional[bool] = False) -> None:
        if utils.file.exists(config.env("BAG3D_INDEX_FILENAME")) and not reconstruct:
            super().__init__(gpd.read_file(config.env("BAG3D_INDEX_FILENAME")))
        else:
            super().__init__(_reconstruct(), crs=os.environ["DEFAULT_CRS"])


def _download_data() -> utils.type.BAG3DTileIndexJSON:
    path = f"{os.environ['BAG3D_WFS_BASE_URL']}&version={os.environ['BAG3D_WFS_VERSION']}&request={os.environ['BAG3D_WFS_REQUEST']}&typeNames={os.environ['BAG3D_WFS_TYPENAMES']}&outputFormat={os.environ['BAG3D_WFS_OUTPUT_FORMAT']}"

    # NOTE - This download should be blocking because the 3DBAG index is required to operate tile by tile.
    response = requests.get(path)
    response.raise_for_status()
    return response.json()


def _parse_data(js: utils.type.BAG3DTileIndexJSON) -> utils.type.BAG3DTileIndexData:
    data = config.default_data_dict()
    for tile in js[os.environ["BAG3D_TILE_FEATURES"]]:
        _pase_tile(tile, data)
    return data


def _reconstruct() -> utils.type.BAG3DTileIndexData:
    return _parse_data(_download_data())


def _parse_tile_id(tile: utils.type.BAG3TileData) -> str:
    return _TILE_ID_FMT.sub("", tile[os.environ["DEFAULT_ID_FIELD_NAME"]]).replace("/",
                                                                                   "-")


def _parse_tile_gm(tile: utils.type.BAG3TileData) -> shapely.Polygon:
    return shapely.Polygon(
        tile[os.environ["DEFAULT_GM_FIELD_NAME"]][os.environ["BAG3D_TILE_COORDINATES"]][
            0])


def _pase_tile(tile: utils.type.BAG3TileData,
               data: utils.type.BAG3DTileIndexData) -> None:
    data[os.environ["DEFAULT_ID_FIELD_NAME"]].append(_parse_tile_id(tile))
    data[os.environ["DEFAULT_GM_FIELD_NAME"]].append(_parse_tile_gm(tile))
