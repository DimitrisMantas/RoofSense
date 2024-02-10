from __future__ import annotations

from os import PathLike

import config
import preprocessing
import utils.iris


# noinspection PyUnusedLocal
def generate_pretraining_data(size: int = 10, background_cutoff: float = 0.5) -> None:
    """Entry point for 'roofsense --sample <size>'."""

    # Initialize the program runtime.
    config.config(pretraining=True)

    # Initialize the data downloaders.
    bag3d_downloader = preprocessing.downloaders.BAG3DDownloader()
    asset_downloader = preprocessing.downloaders.AssetDownloader()

    # Initialize the data parsers.
    bag3d_parser = preprocessing.parsers.BAG3DParser()
    image_parser = preprocessing.parsers.ImageParser()
    lidar_parser = preprocessing.parsers.LiDARParser()

    # Fake a random sample.
    samples = preprocessing.sampler.BAG3DSampler().sample(size)
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
        preprocessing.merger.RasterStackBuilder().merge(sample)

        # Prepare the stacks for annotation.
        preprocessing.splitter.split(sample, background_cutoff)

    # Create the corresponding IRIS configuration file.
    utils.iris.generate_config_f()


def train(root: str | PathLike) -> None:
    """Entry point for 'roofsense --train <root>'."""

    utils.iris.georeference_masks(root)


if __name__ == "__main__":
    generate_pretraining_data()
