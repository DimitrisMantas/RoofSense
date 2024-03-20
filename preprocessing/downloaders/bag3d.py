from __future__ import annotations

import os

import geopandas as gpd
import requests

import config
import utils
from preprocessing.downloaders.base import DataDownloader


class BAG3DDownloader(DataDownloader):
    """Convenience class to download 3DBAG data."""

    def __init__(self) -> None:
        """Initialize the downloader.

        .. note::
            This operation requires the 3DBAG tile index to be present on disc,
            at the location specified by the `BAG3D_TILE_INDEX` configuration parameter.
        """
        self._index: gpd.GeoDataFrame = gpd.read_file(config.env("BAG3D_SHEET_INDEX"))

    def download(self, tile_id: str) -> None:
        """Download a single 3DBAG tile to disk.

        :param tile_id: The tile ID (e.g., 9-284-556).

        .. note::
            Tiles are downloaded in CityJSON format.

            Tiles are saved on disk, in the directory specified by the `TEMP_DIR`
            configuration parameter, and named using their ID (e.g.,
            `9-284-556.city.json`).
        """
        try:
            url: str = self._index.loc[self._index.tid == tile_id].url.iat[0]
        except IndexError:
            msg = (
                f"Failed to identify 3DBAG tile with ID: {tile_id!r}. Tile does not "
                f"exist."
            )
            raise IndexError(msg)
        dst_filepath = os.path.join(config.env("TEMP_DIR"), tile_id + ".city.json")

        with requests.Session() as session:
            utils.file.BlockingFileDownloader(
                url,
                filename=dst_filepath,
                session=session,
            ).download()


if __name__ == "__main__":
    config.config()

    downloader = BAG3DDownloader()
    # Test with a valid tile ID.
    downloader.download("9-284-556")
    # Test with an invalid tile ID.
    try:
        downloader.download("")
    except IndexError:
        pass
