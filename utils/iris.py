from __future__ import annotations

import json
import pathlib

import config
import utils.file


def write_cfg():
    with pathlib.Path(config.var("IRIS_BASE_CFG")).open() as f:
        cfg = json.load(f)
    out_path = (
        f"{config.env('PRETRAINING_DATA_DIR')}"
        f"{config.var('IRIS_CFG_NAME')}"
        f"{config.var('JSON')}"
    )
    if utils.file.exists(out_path):
        return
    img_shp = [config.var("BLOCK_SIZE"), config.var("BLOCK_SIZE")]
    cfg["images"]["shape"] = img_shp
    cfg["segmentation"]["mask_area"] = [0, 0, *img_shp]
    with pathlib.Path(out_path).open("w") as f:
        json.dump(cfg, f)
