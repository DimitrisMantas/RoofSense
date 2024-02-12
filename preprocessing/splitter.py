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
    # Dissolve the surfaces so that only their edges are buffered.
    surfs = utils.geom.read_surfaces(obj_id).dissolve()

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
        original_surf_mask, *_ = rasterio.mask.raster_geometry_mask(stack,
            shapes=surfs[config.var("DEFAULT_GM_FIELD_NAME")])
        buffered_surf_mask, *_ = rasterio.mask.raster_geometry_mask(stack,
            shapes=surfs[config.var("DEFAULT_GM_FIELD_NAME")].buffer(float(config.var(
                "BUFFER_DISTANCE"))), )

        block: rasterio.windows.Window
        # Use the data blocks of the first band.
        # NOTE: This ensures indexing notation ensures that all bands share the
        #       same internal data block structure.
        for (row, col), block in stack.block_windows(-1):
            if block.width != block.height:
                continue

            # TODO: Document this block.
            original_patch_data = stack.read(window=block, masked=True)
            buffered_patch_data = original_patch_data.copy()
            original_patch_data.mask = (original_patch_data.mask | original_surf_mask[
                                                                   block.row_off: block.row_off + block.height,
                                                                   block.col_off: block.col_off + block.width, ])
            original_patch_data = original_patch_data.filled(0)
            buffered_patch_data.mask = (buffered_patch_data.mask | buffered_surf_mask[
                                                                   block.row_off: block.row_off + block.height,
                                                                   block.col_off: block.col_off + block.width, ])
            buffered_patch_data = buffered_patch_data.filled(0)

            act_zeros = original_patch_data.size - np.count_nonzero(original_patch_data)
            max_zeros = original_patch_data.size * background_cutoff
            if act_zeros > max_zeros:
                continue

            patch_meta = stack.meta
            patch_meta.update(width=block.width,
                height=block.height,
                transform=rasterio.windows.transform(block, stack.transform),
                # Disable tiling since it is no longer required.
                tiled=False,
            )

            original_patch_path = pathlib.Path(f"{config.env('ORIGINAL_DATA_DIR')}"
                                               f"{config.var('TRAINING_IMAG_DIRNAME')}").joinpath(
                f"{stack_path.stem.replace(config.var('RASTER_STACK'), '')}"
                f"{config.var('SEPARATOR')}"
                f"{row}"
                f"{config.var('SEPARATOR')}"
                f"{col}"
                f"{stack_path.suffix}")
            original_patch: rasterio.io.DatasetWriter
            with rasterio.open(original_patch_path,
                    "w",
                    **patch_meta) as original_patch:
                original_patch.write(original_patch_data)

            buffered_patch_path = pathlib.Path(f"{config.env('BUFFERED_DATA_DIR')}"
                                               f"{config.var('TRAINING_IMAG_DIRNAME')}").joinpath(
                f"{stack_path.stem.replace(config.var('RASTER_STACK'), '')}"
                f"{config.var('SEPARATOR')}"
                f"{row}"
                f"{config.var('SEPARATOR')}"
                f"{col}"
                f"{stack_path.suffix}")
            buffered_patch: rasterio.io.DatasetWriter
            with rasterio.open(buffered_patch_path,
                    "w",
                    **patch_meta) as buffered_patch:
                buffered_patch.write(buffered_patch_data)
