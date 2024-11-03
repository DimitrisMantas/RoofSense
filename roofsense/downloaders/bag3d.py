from __future__ import annotations

import os

import geopandas as gpd
import requests
from typing_extensions import override

from roofsense import config
from roofsense.downloaders._base import _Downloader
from roofsense.utils.file import BlockingFileDownloader


class BAG3DDownloader(_Downloader):
    """Convenience class for downloading 3DBAG tiles."""

    def __init__(self) -> None:
        """Initialize the downloader.

        .. note::
            This operation requires that the 3DBAG tile index is present at the
            location specified by the ``BAG3D_TILE_INDEX`` configuration parameter.
        """
        self._index: gpd.GeoDataFrame = gpd.read_file(config.env("BAG3D_SHEET_INDEX"))

    @override
    def download(self, tile_id: str) -> None:
        """Download a single 3DBAG tile.

        :param tile_id: The tile ID (e.g., ``"9-284-556"``).

        .. note::
            Tiles are downloaded in the `CityJSON <https://www.cityjson.org/>`_ file
            format.

            Tiles are saved in the directory specified by the ``TEMP_DIR``
            configuration parameter, and named using their ID (e.g.,
            ``9-284-556.city.json``).
        """
        try:
            url: str = self._index.loc[self._index.tid == tile_id].url.iat[0]
        except IndexError:
            msg = (
                f"Failed to identify 3DBAG tile with ID: {tile_id!r}. Tile does not "
                f"exist."
            )
            raise IndexError(msg)
        # TODO: Think of a better way to compile paths from individual user-defined
        #  parameters.
        dst_filepath = os.path.join(config.env("TEMP_DIR"), tile_id + ".city.json")

        with requests.Session() as session:
            BlockingFileDownloader(
                url, filename=dst_filepath, session=session
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
