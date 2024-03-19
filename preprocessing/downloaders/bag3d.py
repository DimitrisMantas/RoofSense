from __future__ import annotations

import pathlib

import geopandas as gpd
import requests

import config
import utils


class BAG3DDownloader:
    def __init__(self) -> None:
        self._index: gpd.GeoDataFrame = gpd.read_file(config.env("BAG3D_SHEET_INDEX"))

    def download(self, tile_id: str) -> None:
        try:
            url: str = self._index.loc[self._index.tid == tile_id].url.iat[0]
        except IndexError:
            msg = (
                f"Failed to identify 3DBAG tile with ID: {tile_id!r}. Tile does not "
                f"exist."
            )
            raise IndexError(msg)
        dst_filepath = pathlib.Path(config.env("TEMP_DIR")) / pathlib.Path(url).name

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
