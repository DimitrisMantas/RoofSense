import os

import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp

from roofsense.bag3d import BAG3DTileStore
from roofsense.utilities import raster
from roofsense.utilities.file import confirm_write_op


class RasterStackBuilder:
    def __init__(self, store: BAG3DTileStore) -> None:
        self.dirpath = store.dirpath

    def merge(self, tile_id: str, overwrite: bool = False) -> None:
        out_path = os.path.join(self.dirpath, f"{tile_id}.stack.tif")
        if not confirm_write_op(out_path, overwrite=overwrite):
            return

        in_paths = [
            os.path.join(self.dirpath, f"{tile_id}{img_tp}.tif")
            for img_tp in [".rgb", ".rfl", ".slp", ".ndsm", ".den"]
        ]

        out_meta = raster.DefaultProfile(dtype=np.float32, nodata=np.nan)
        # NOTE: The RGB image contains 3 bands.
        out_meta.update(count=len(in_paths) + 2)

        rgb: rasterio.io.DatasetReader
        with rasterio.open(in_paths[0]) as rgb:
            out_meta.update(
                crs=rgb.crs, width=rgb.width, height=rgb.height, transform=rgb.transform
            )

            out_data = np.full((7, rgb.height, rgb.width), fill_value=np.nan)
            out_data[:3, ...] = rgb.read()

        stack: rasterio.io.DatasetWriter
        with rasterio.open(out_path, mode="w", **out_meta) as stack:
            for band_id, path in enumerate(in_paths[1:-1], start=3):
                tmp: rasterio.io.DatasetReader
                with rasterio.open(path) as tmp:
                    tmp_data: np.ndarray = tmp.read(
                        indexes=1,
                        out_shape=(out_meta["height"], out_meta["width"]),
                        resampling=rasterio.enums.Resampling.bilinear,
                    )

                    # FIXME: Should this be applied when generating the band?
                    #   - I think it's fine here because if we bandlimit before
                    #     this step object edges become less pronounced.
                    # FIXME: Should this be applied to the nDRM band as well?
                    # dsm
                    if band_id == 5:
                        tmp_data = tmp_data.clip(
                            min=np.percentile(tmp_data, 2),
                            max=np.percentile(tmp_data, 98),
                        )
                    out_data[band_id, ...] = tmp_data

            # ------------------------------
            # Parse Density
            with rasterio.open(in_paths[-1]) as tmp:
                original_sum = tmp.read().sum()

            with rasterio.open(in_paths[-1]) as tmp:
                tmp_data: np.ndarray = tmp.read(
                    indexes=1,
                    out_shape=(out_meta["height"], out_meta["width"]),
                    resampling=rasterio.enums.Resampling.nearest,
                )

            # Resample
            tmp_data = tmp_data * (original_sum / tmp_data.sum())

            # Clip
            tmp_data = tmp_data.clip(max=np.percentile(tmp_data, 98))

            # Merge
            out_data[band_id + 1, ...] = tmp_data
            # ------------------------------

            stack.write(out_data)

            # TODO: Mask the stack here
