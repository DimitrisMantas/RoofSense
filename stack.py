from abc import abstractmethod
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
from overrides import override

import config
import utils
from utils import raster


class DataMerger:
    def __init__(self) -> None:
        self._surfs: Optional[gpd.GeoDataFrame] = None

    @abstractmethod
    def merge(self, obj_id: str) -> None: ...


class RasterStackBuilder(DataMerger):
    def __init__(self) -> None:
        super().__init__()

    @override
    def merge(self, obj_id: str) -> None:
        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RASTER_STACK')}"
            f"{config.var('TIF')}"
        )

        if utils.file.exists(out_path):
            return

        in_paths = [
            f"{config.env('TEMP_DIR')}{obj_id}{img_tp}{config.var('TIF')}"
            for img_tp in [
                config.var("RGB"),
                config.var("REFLECTANCE"),
                config.var("SLOPE"),
                ".ndsm",
                ".den",
            ]
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
            for band_id, path in enumerate(in_paths[1:], start=3):
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
                            min=np.percentile(tmp_data, 1),
                            max=np.percentile(tmp_data, 99),
                        )
                    # density
                    if band_id == 6:
                        # limit erroneously high density values along building edges.
                        tmp_data = tmp_data.clip(max=np.percentile(tmp_data, 99))

                    out_data[band_id, ...] = tmp_data
            stack.write(out_data)

            # TODO: Mask the stack here
