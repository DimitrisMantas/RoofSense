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
