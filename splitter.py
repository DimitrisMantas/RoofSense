import os.path
import pathlib

import numpy as np
import rasterio
import rasterio.mask
import rasterio.plot
import rasterio.windows

from roofsense.annotation.exporters import to_png
from roofsense.bag3d import BAG3DTileStore, LevelOfDetail
from roofsense.utils.file import confirm_write_op


def split(
    store: BAG3DTileStore,
    tile_id: str,
    background_cutoff: float,
    limit: int,
    overwrite: bool = False,
) -> int:
    surfs = store.read_tile(
        tile_id, lod=LevelOfDetail.LoD22
    ).dissolve()  # todo: does this speed up masking?

    rng = np.random.default_rng(seed=0)

    num_local_processed_chips = 0
    num_total_processed_chips = 0

    stack_path = os.path.join(store.dirpath, f"{tile_id}.stack.tif")
    stack: rasterio.io.DatasetReader
    with rasterio.open(stack_path) as stack:
        stack_data, *_ = rasterio.mask.mask(
            stack, shapes=surfs["geometry"], filled=False
        )

        win: rasterio.windows.Window
        # Use the data blocks of the first band.
        # NOTE: This ensures indexing notation ensures that all bands share the
        #       same internal data chip structure.
        for (row, col), win in stack.block_windows(-1):
            if num_total_processed_chips == limit:
                return limit

            if win.width != win.height:
                continue

            if rng.binomial(1, p=0.5) == 0:
                continue

            if num_local_processed_chips == 3:
                num_local_processed_chips = 0
                continue

            num_local_processed_chips += 1

            chip_path = os.path.join(
                "dataset",
                "imgs",
                f"{pathlib.Path(stack_path).stem.replace('.stack', '')}_{row}_{col}.tif",
            )

            if not confirm_write_op(chip_path, overwrite=overwrite):
                num_total_processed_chips += 1
                continue

            chip_data = stack_data[
                :,
                win.row_off : win.row_off + win.height,
                win.col_off : win.col_off + win.width,
            ]

            num_band_cells = chip_data.shape[1] * chip_data.shape[2]
            num_back_cells = chip_data.mask[0, ...].sum()
            if num_back_cells > num_band_cells * background_cutoff:
                continue

            chip_data = chip_data.filled(0)

            chip_meta = stack.meta
            chip_meta.update(
                width=win.width,
                height=win.height,
                transform=rasterio.windows.transform(win, stack.transform),
                # Disable tiling since it is no longer required.
                tiled=False,
            )

            chip: rasterio.io.DatasetWriter
            with rasterio.open(chip_path, "w", **chip_meta) as chip:
                chip.write(chip_data)

            png_path = os.path.join(
                "dataset",
                "chps",
                f"{pathlib.Path(stack_path).stem.replace('.stack', '')}_{row}_{col}.tif",
            )
            if confirm_write_op(png_path, overwrite=overwrite):
                # TODO: expose exporter to user
                to_png(chip_data, png_path)

            num_total_processed_chips += 1
        return num_total_processed_chips
