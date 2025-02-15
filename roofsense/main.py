import os

import geopandas as gpd

from roofsense.preprocessing.chip_samplers import BAG3DSampler


def enable_speedups():
    # Enable GeoPandas speed-ups.
    gpd.options.io_engine = "pyogrio"
    os.environ.update({"PYOGRIO_USE_ARROW": "1"})

    # See https://github.com/pangeo-data/cog-best-practices for more information.
    os.environ.update(
        {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "AWS_NO_SIGN_REQUEST": "YES",
            "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
            "GDAL_SWATH_SIZE": "200000000",
            "VSI_CURL_CACHE_SIZE": "200000000",
        }
    )


def generate_pretraining_data(size: int = 300, background_cutoff: float = 0.8) -> None:
    # Fake a random sample.
    BAG3DSampler(
        seeds_filepath=r"data/cities.gpkg",
        image_index_filepath=r"data/index/image/image.gpkg",
        lidar_index_filepath=r"data/index/lidar/lidar.gpkg",
    ).sample(size, background_cutoff)


if __name__ == "__main__":
    enable_speedups()
    generate_pretraining_data()
