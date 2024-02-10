from __future__ import annotations

import warnings
from os import PathLike

import config
import preprocessing
import utils.iris


# noinspection PyUnusedLocal
def generate_training_data(size: int = 10, background_cutoff: float = 0.5) -> None:
    """
    The entry point for:
        roofsense --gen-training-data <size> --background-cutoff <pct>
    """

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
    for i, sample in enumerate(samples, start=1):
        print(f"Generating training data for tile {i}/{size}: {sample}...")

        # Download the corresponding 3DBAG data.
        bag3d_downloader.download(sample)
        # Parse the data.
        bag3d_parser.parse(sample)

        # Check if the spatial distribution of the buildings in the tile is erroneous
        # (e.g., small disjoint building clusters ets.).
        # NOTE: This ensures that only tiles with the minimal possible number of
        # assets are processed in subsequent steps.
        surfs = utils.geom.read_surfaces(sample)

        surf_bbox = surfs.total_bounds
        bbox_xlen = surf_bbox[2] - surf_bbox[0]
        bbox_ylen = surf_bbox[3] - surf_bbox[1]
        # TODO: Remove this check once the point cloud rasterisation process has been
        #      optimised.
        # NOTE: Erroneous tiles are defined as those whose axis-aligned bounding box
        #       has at least one side
        #       which is longer than the maximum side length of an AHN4 tile.
        if bbox_xlen > 1250 or bbox_ylen > 1250:
            warnings.warn("The tile is erroneous. Skipping...")
            continue

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
    """
    The entry point for:
        roofsense --train <cfg>
    """

    utils.iris.georeference_masks(root)


if __name__ == "__main__":
    generate_training_data()
