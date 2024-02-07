from __future__ import annotations

import pathlib

import rasterio
import rasterio.windows

import config


def split(obj_id: str) -> None:
    stack = pathlib.Path(
        (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RASTER_STACK')}"
            f"{config.var('TIF')}"
        )
    )

    # for path in src_paths:
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
