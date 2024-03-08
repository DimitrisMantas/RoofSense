from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import rasterio.io
import rasterio.mask
import scipy as sp

import config
import utils.file


def generate_configuration_file() -> None:
    with pathlib.Path(config.var("IRIS_BASE_CFG")).open() as f:
        cfg = json.load(f)

    cfg_path = (f"{config.env('TRAINING_DATA_DIR')}"
        f"{config.var('IRIS_CFG_NAME')}"
        f"{config.var('JSON')}"
    )
    if utils.file.exists(cfg_path):
        return

    img_shp = [int(config.var("BLOCK_SIZE")), int(config.var("BLOCK_SIZE"))]
    cfg["images"]["shape"] = img_shp
    cfg["segmentation"]["mask_area"] = [0, 0, *img_shp]

    with pathlib.Path(cfg_path).open("w") as f:
        json.dump(cfg, f)


def postprocess_masks() -> None:
    """
    1. Add a buffer around the buildings corresponding to the roof surfaces extracted
       from the 3DBAG to avoid oversmoothing and label it as invalid.
    2. Reannotate the area on the exterior of the buffered geometry using the __ignore__
       label.
    3. Pass the resulting mask through a median filter to eliminate noise.
       # TODO: Figure out how to handle the edges so that the effect buffers from
               neighboring buildings that intersect the image is minimized.
    """
    original_img_paths = sorted(pathlib.Path(config.env("ORIGINAL_DATA_DIR")).joinpath(
        "imgs").glob("*.tif"))
    original_msk_paths = sorted(pathlib.Path(config.env("ORIGINAL_DATA_DIR")).joinpath(
        "msks").glob("*.tif"))

    buffered_msk_paths = _georeference_masks(original_img_paths, original_msk_paths)
    buffer_buildings(buffered_msk_paths)


def buffer_buildings(msk_paths) -> None:
    """
    Buffer the valid segmentation area around buildings to minimize potential
    oversmoothing effects along their edges.

    :param msk_paths:
    :return:
    """
    surfs = (utils.geom.read_surfaces("9-284-556").dissolve()["geometry"].buffer(float(
        config.var("BUFFER_DISTANCE"))))
    for path in msk_paths:
        with rasterio.open(path) as f:
            out_meta = f.profile
            out_data, _ = rasterio.mask.mask(f, shapes=surfs, nodata=9, indexes=1)  # out_data = sp.signal.medfilt2d(out_data)
        with rasterio.open(path, "w", **out_meta) as f:
            f.write(out_data, indexes=1)


SegmentationMask = np.ndarray[[Any, Any], np.uint8]


def denoise(mask: SegmentationMask, kernel_size: int) -> SegmentationMask:
    return sp.signal.medfilt2d(mask, kernel_size=kernel_size)


def _georeference_masks(img_paths, msk_paths):
    """Georeference the annotation masks."""
    paths = []
    for img_path, msk_path in zip(img_paths, msk_paths):
        img: rasterio.io.DatasetReader
        with rasterio.open(img_path) as img:
            img_crs = img.crs
            img_trf = img.transform
        msk: rasterio.io.DatasetWriter
        with rasterio.open(msk_path) as msk:
            msk_data = msk.read()
            msk_meta = msk.meta
        msk_meta.update(crs=img_crs, transform=img_trf)
        path = str(msk_path).replace("original", "buffered")
        paths.append(path)
        with rasterio.open(path, "w", **msk_meta) as out:
            out.write(msk_data)
    return paths
