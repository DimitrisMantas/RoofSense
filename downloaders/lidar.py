import urllib.parse

import geopandas as gpd
import requests

import config
import utils.file

# TODO: Reformat, finalize function and variable names, and add documentation.
# TODO: Write a generic 3DBAG asset downloader because this downloader is more 90% the
#       same as `image`.

BASE_URL = "https://geotiles.citg.tudelft.nl/AHN4_T/"


def load_index() -> gpd.GeoDataFrame:
    return gpd.read_file(config.env("AHN34_INDEX_FILENAME"))


def download(obj_id: str, index: gpd.GeoDataFrame) -> None:
    obj_path = (
        f"{config.env('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
        f"{config.var('GEOPACKAGE')}"
    )
    obj_bbox = gpd.read_file(obj_path)
    obj_bbox["geometry"] = obj_bbox["geometry"].buffer(
        float(config.var("BUFFER_DISTANCE"))
    )

    # Fetch the image IDs to download.
    img_ids = index.overlay(obj_bbox)["lidar_id"].unique()

    # Build the image web addresses and local names.
    # TOSELF: There has to be a way to clean up this block using `itertools`?!?
    img_addrs = [f"{BASE_URL}" f"{id_}" f"{config.var('LAZ')}" for id_ in img_ids]

    img_names = [
        (
            f"{config.var('TEMP_DIR')}"
            f"{urllib.parse.urlparse(addr).path.rsplit('/')[-1]}"
        )
        for addr in img_addrs
    ]

    with requests.Session() as s:
        utils.file.ThreadedFileDownloader(img_addrs, img_names, session=s).download()
