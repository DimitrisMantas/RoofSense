import urllib.parse

import requests

import config
import utils
from downloaders._base import DataDownloader
from utils.file import ThreadedFileDownloader

# TODO: Promote this static constant to an environment variable.
_BASE_DATA_ADDRESS = "https://geotiles.citg.tudelft.nl/AHN4_T/"


class LiDARDataDownloader(DataDownloader):
    def __init__(self) -> None:
        super().__init__(config.env("ASSET_INDEX_FILENAME"))

    # TODO: Find out why the override decorator cannot be imported from the typing
    #       module.
    # TODO: Find out if there's more code than can be factored out between the image
    #       and LiDAR data downloaders. @override
    def download(self, obj_id: str) -> None:
        surf = utils.geom.read(obj_id)
        buff = utils.geom.buffer(surf)

        # TODO: Harmonize these variable names.
        data_id = self._sindex.overlay(buff)["lidar_id"].unique()
        request = [f"{_BASE_DATA_ADDRESS}{i}{config.var('LAZ')}" for i in data_id]
        pathnam = [
            (
                f"{config.var('TEMP_DIR')}"
                f"{obj_id}."
                f"{urllib.parse.urlparse(i).path.rsplit('/')[-1]}"
            )
            for i in request
        ]

        with requests.Session() as s:
            ThreadedFileDownloader(request, pathnam, session=s).download()
