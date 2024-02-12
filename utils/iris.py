from __future__ import annotations

import json
import pathlib
from os import PathLike

import rasterio.io

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


def postprocess_masks(root_dir: str | PathLike) -> None:
    """
    1. Add a buffer around the buildings corresponding to the roof surfaces extracted
       from the 3DBAG to avoid oversmoothing and label it as invalid.
    2. Reannotate the area on the exterior of the buffered geometry using the __ignore__
       label.
    3. Pass the resulting mask through a median filter to eliminate noise.
       # TODO: Figure out how to handle the edges so that the effect buffers from
               neighboring buildings that intersect the image is minimized.
    """
    pass


def georeference_masks(root_dir: str | PathLike) -> None:
    """Georeference the annotation masks."""
    img_paths = sorted(pathlib.Path(root_dir).joinpath("imgs").glob("*.tif"))
    msk_paths = sorted(pathlib.Path(root_dir).joinpath("msks").glob("*.tif"))
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
        with rasterio.open(msk_path, "w", **msk_meta) as out:
            out.write(msk_data)
