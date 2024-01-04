import urllib.parse

import geopandas as gpd
import requests

import config
import utils.file

# TODO: Reformat, finalize function and variable names, and add documentation.
# TODO: Write a generic 3DBAG asset downloader because this downloader is more 90% the
#       same as `lidar`.

BASE_URL = "https://geotiles.citg.tudelft.nl/Luchtfoto_2023/"


def load_index() -> gpd.GeoDataFrame:
    return gpd.read_file(config.env("ORTHO_INDEX_FILENAME"))


def download(obj_id: str, index: gpd.GeoDataFrame) -> None:
    obj_path = (
        f"{config.env('TEMP_DIR')}"
        f"{obj_id}"
        f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
        f"{config.var('GEOPACKAGE')}"
    )
    obj_bbox = gpd.read_file(obj_path)
    obj_bbox["geometry"] = obj_bbox["geometry"].buffer(10)

    # Fetch the image IDs to download.
    img_ids = index.overlay(obj_bbox)["id_1"].unique()

    # Build the image web addresses and local names.
    # TOSELF: There has to be a way to clean up this block using `itertools`?!?
    img_addrs = [
        f"{BASE_URL}" f"{tp}" f"_" f"{id_}" f"{config.var('TIFF')}"
        for id_ in img_ids
        for tp in ["RGB", "CIR"]
    ]

    img_names = [
        (
            f"{config.var('TEMP_DIR')}"
            f"{urllib.parse.urlparse(addr).path.rsplit('/')[-1]}"
        )
        for addr in img_addrs
    ]

    with requests.Session() as s:
        utils.file.ThreadedFileDownloader(img_addrs, img_names, session=s).download()
