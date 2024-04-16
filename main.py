from __future__ import annotations

import common
import config
import training


def generate_pretraining_data(size: int = 10, background_cutoff: float = 0.7) -> None:
    """
    Entry point for:
        roofsense --gen-pretrain-data <size> --bg-cutoff <pct>
    """
    # Initialize the program runtime.
    config.config(training=True)

    # Initialize the data downloaders.
    bag3d_downloader = common.downloaders.BAG3DDownloader()
    asset_downloader = common.downloaders.AssetDownloader()

    # Initialize the data parsers.
    bag3d_parser = common.parsers.BAG3DParser()
    image_parser = common.parsers.ImageParser()
    lidar_parser = common.parsers.LiDARParser()

    # Fake a random sample.
    samples = common.sampler.BAG3DSampler().sample(size)
    for sample in samples:
        # Download the corresponding 3DBAG data.
        bag3d_downloader.download(sample)
        # Parse the data.
        bag3d_parser.parse(sample)

        # Download the corresponding assets.
        asset_downloader.download(sample)
        # Parse the data.
        image_parser.parse(sample)
        lidar_parser.parse(sample)

        # Create the raster stack.
        common.merger.RasterStackBuilder().merge(sample)

        # Prepare the stacks for annotation.
        training.splitter.split(sample, background_cutoff)


def train() -> None:
    """
    Entry point for:
        roofsense --train
    """
    ...


if __name__ == "__main__":
    generate_pretraining_data()
