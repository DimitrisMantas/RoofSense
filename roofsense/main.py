from __future__ import annotations

import config
from roofsense.downloaders.asset import AssetDownloader
from roofsense.downloaders.bag3d import BAG3DDownloader
from roofsense.parsers.bag3d import BAG3DParser
from roofsense.parsers.image import ImageParser
from roofsense.parsers.lidar import LiDARParser
from stack import RasterStackBuilder


def generate_pretraining_data(size: int = 300, background_cutoff: float = 0.8) -> None:
    """Entry point for:
    roofsense --gen-pretrain-data <size> --bg-cutoff <pct>
    """
    # Initialize the program runtime.
    config.config(training=True)
    # Fake a random sample.
    # samples = training.sampler.BAG3DSampler().sample(size, background_cutoff)

    tile_ids = [
        # "10-284-560",
        "10-280-560",
        # "10-282-560",
        # "10-286-560",
        # "9-280-552",
        # "9-284-552",
        # "9-280-556",
        # "8-288-552",
        # "10-148-336"  # test
    ]
    for tile_id in tile_ids:
        # Initialize the data downloaders.
        bag3d_downloader = BAG3DDownloader()
        asset_downloader = AssetDownloader()

        # Initialize the data parsers.
        bag3d_parser = BAG3DParser()
        image_parser = ImageParser(dirpath=config.env("TEMP_DIR"))
        lidar_parser = LiDARParser(dirpath=config.env("TEMP_DIR"))
        # Download the corresponding 3DBAG data.
        bag3d_downloader.download(tile_id)
        # Parse the data.
        bag3d_parser.parse(tile_id)

        # Download the corresponding assets.
        asset_downloader.download(tile_id)
        # Parse the data.
        image_parser.parse(tile_id)
        lidar_parser.parse(tile_id)

        # Create the raster stack.
        RasterStackBuilder().merge(tile_id)


def train() -> None:
    """Entry point for:
    roofsense --train
    """
    ...


if __name__ == "__main__":
    generate_pretraining_data()
