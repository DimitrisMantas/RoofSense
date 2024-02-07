from __future__ import annotations

import json
import pathlib
from os import PathLike

import rasterio.io

import config
import utils.file


def generate_config_f():
    with pathlib.Path(config.var("IRIS_BASE_CFG")).open() as f:
        cfg = json.load(f)
    out_path = (
        f"{config.env('PRETRAINING_DATA_DIR')}"
        f"{config.var('IRIS_CFG_NAME')}"
        f"{config.var('JSON')}"
    )
    if utils.file.exists(out_path):
        return
    img_shp = [int(config.var("BLOCK_SIZE")), int(config.var("BLOCK_SIZE"))]
    cfg["images"]["shape"] = img_shp
    cfg["segmentation"]["mask_area"] = [0, 0, *img_shp]
    with pathlib.Path(out_path).open("w") as f:
        json.dump(cfg, f)


def georeference_masks(root_dir: str | PathLike) -> None:
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
