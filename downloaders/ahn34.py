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

import geopandas as gpd
import requests

import config
import utils.file

# TODO - Reformat, finalize function and variable names, and add documentation.

BASE_URL = "https://geotiles.citg.tudelft.nl/AHN4_T/"


def load_index() -> gpd.GeoDataFrame:
    return gpd.read_file(config.env("AHN34_INDEX_FILENAME"))


def download(id_: str, index: gpd.GeoDataFrame) -> None:
    path = f"{config.env('TEMP_DIR')}{id_}{config.var('DEFAULT_BUILDING_FOOTPRINT_FILE_ID')}{config.var('GEOPACKAGE')}"
    building_footprints = gpd.read_file(path)

    ids = index.overlay(building_footprints)["id_1"].unique()

    urls = [f"{BASE_URL}{id_}{config.var('LAZ')}" for id_ in ids]
    filenames = [f"{config.var('TEMP_DIR')}{id_}{config.var('LAZ')}" for id_ in ids]
    with requests.Session() as session:
        utils.file.ThreadedFileDownloader(urls, filenames, session=session).download()
