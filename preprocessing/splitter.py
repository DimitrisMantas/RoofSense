from __future__ import annotations

import pathlib

import numpy as np
import rasterio
import rasterio.mask
import rasterio.windows

import config
import utils.geom

config.config()


def split(obj_id: str, background_cutoff: float) -> None:
    surfs = utils.geom.read_surfaces(obj_id)

    stack_path = pathlib.Path(
        (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RASTER_STACK')}"
            f"{config.var('TIF')}"
        )
    )
    stack: rasterio.io.DatasetReader
    with rasterio.open(stack_path) as stack:
        surf_mask, *_ = rasterio.mask.raster_geometry_mask(stack,
            shapes=surfs[config.var("DEFAULT_GM_FIELD_NAME")])
        block: rasterio.windows.Window
        # Use the data blocks of the first band.
        # NOTE: This ensures indexing notation ensures that all bands share the
        #       same internal data block structure.
        for (row, col), block in stack.block_windows(-1):
            if block.width != block.height:
                continue

            # TODO: Document this block.
            patch_data = stack.read(window=block, masked=True)
            patch_data.mask = (patch_data.mask | surf_mask[
                                                 block.row_off: block.row_off + block.height,
                                                 block.col_off: block.col_off + block.width, ])
            patch_data = patch_data.filled(0)

            act_zeros = patch_data.size - np.count_nonzero(patch_data)
            max_zeros = patch_data.size * background_cutoff
            if act_zeros > max_zeros:
                continue

            patch_meta = stack.meta
            patch_meta.update(width=block.width,
                height=block.height,
                transform=rasterio.windows.transform(block, stack.transform),
                # Disable tiling since it is no longer required.
                tiled=False,
            )

            out_path = pathlib.Path(
                f"{config.env('PRETRAINING_DATA_DIR')}imgs").joinpath(f"{stack_path.stem}_{row}_{col}{stack_path.suffix}")
            patch: rasterio.io.DatasetWriter
            with rasterio.open(out_path, "w", **patch_meta) as patch:
                patch.write(patch_data)
