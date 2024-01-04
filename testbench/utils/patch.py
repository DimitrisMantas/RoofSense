from __future__ import annotations

import glob
import itertools
import math
import os
import random
from typing import Generator

import rasterio
import rasterio.windows

random.seed(42)


def patch(filepath: str, pathname: str, res: int = 128, sample: float = None) -> None:
    with rasterio.open(filepath) as src:
        profile = src.meta.copy()
        if sample is not None:
            num_all_patches = count_patches(src, res)
            num_ret_patches = math.ceil(num_all_patches * sample)
            samples = random.sample(range(num_all_patches), num_ret_patches)

            patches = sample_patches(src, res, samples)
        else:
            patches = gather_patches(src, res)
        for k, window, transform in patches:
            profile["width"], profile["height"] = window.width, window.height
            profile["transform"] = transform

            img_name = os.path.splitext(os.path.basename(i_img_path))[0]
            o_img_path = os.path.join(pathname, f"{img_name}_{k}.tiff")
            with rasterio.open(o_img_path, "w", **profile) as g:
                g.write(src.read(window=window))


IMG_DIR = "../data/src/msk"
MSK_DIR = "masks"

TARGET_SIZE = 1024

img_query = os.path.join(IMG_DIR, "*.tiff")
msk_query = os.path.join(MSK_DIR, "*.tif")

img_paths = glob.glob(img_query)
msk_paths = glob.glob(msk_query)

img_paths.sort()
msk_paths.sort()


# np.ndarray[tuple[Any, Any] | tuple[Any, Any, Any], float]


def count_patches(src: rasterio.io.DatasetReader, res: int) -> int:
    """
    Compute the number of square patches a single- or multi-band image can be split
    into.

    :param src: The image file descriptor.
    :param res: The side length of a single patch in pixels.

    :return: TODO
    """
    return src.meta["width"] // res * src.meta["height"] // res


def gather_patches(
    src: rasterio.io.DatasetReader, res: int
) -> Generator[tuple[int, rasterio.windows.Window, rasterio.transform.Affine]]:
    """
    Generate the square patches a single- or multi-band image can be split into.

    :param src: The image file descriptor.
    :param res: The side length of a single patch in pixels.

    :return: A single patch and its transform.
    """
    num_cols, num_rows = src.meta["width"], src.meta["height"]
    offsets = itertools.product(range(0, num_cols, res), range(0, num_rows, res))
    extents = rasterio.windows.Window(
        col_off=0, row_off=0, width=num_cols, height=num_rows
    )
    i = 1
    for col_off, row_off in offsets:
        window = rasterio.windows.Window(col_off, row_off, res, res).intersection(
            extents
        )
        if window.width != window.height:
            continue
        transform = rasterio.windows.transform(window, src.transform)
        yield i, window, transform
        i += 1


def sample_patches(
    src: rasterio.io.DatasetReader, res: int, sample: list[int]
) -> Generator[tuple[int, rasterio.windows.Window, rasterio.transform.Affine]]:
    gen = gather_patches(src, res)
    off = 0
    for i in sorted(sample):
        for _ in range((i - off) - 1):
            next(gen)
        yield next(gen)
        off = i


for i, i_img_path in enumerate(img_paths):
    print(i_img_path)

    patch(i_img_path, "../data/msk", res=128)  # chips(i_msk_path, "msk")
