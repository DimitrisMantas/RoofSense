from __future__ import annotations

import pathlib

import numpy as np
import rasterio
import rasterio.windows

import config


def split(obj_id: str, background_cutoff: float) -> None:
    stack = pathlib.Path(
        (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RASTER_STACK')}"
            f"{config.var('TIF')}"
        )
    )
    src: rasterio.io.DatasetReader
    with rasterio.open(stack) as src:
        window: rasterio.windows.Window
        # Use the data blocks of the first band.
        # NOTE: This ensures indexing notation ensures that all bands share the
        #       same internal data block structure.
        for rowcol, window in src.block_windows(-1):
            if window.width != window.height:
                continue
            out_data = src.read(window=window)

            act_zeros = out_data.size - np.count_nonzero(out_data)
            max_zeros = out_data.size * background_cutoff
            if act_zeros > max_zeros:
                continue
            out_meta = src.meta
            out_meta.update(
                width=window.width,
                height=window.width,
                transform=rasterio.windows.transform(window, src.transform),
                # Disable tiling since it is no longer required.
                tiled=False,
            )
            out_path = pathlib.Path(
                f"{config.env('PRETRAINING_DATA_DIR')}imgs"
            ).joinpath(f"{stack.stem}_{rowcol[0]}_{rowcol[1]}{stack.suffix}")
            out: rasterio.io.DatasetWriter
            with rasterio.open(out_path, "w", **out_meta) as out:
                out.write(out_data)
