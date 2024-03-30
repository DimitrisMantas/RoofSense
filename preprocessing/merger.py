from abc import abstractmethod
from typing import Optional

import geopandas as gpd
import rasterio
import rasterio.mask
from overrides import override

import config
import utils.file
from preprocessing.parsers.utils import raster


class DataMerger:
    def __init__(self) -> None:
        self._surfs: Optional[gpd.GeoDataFrame] = None

    @abstractmethod
    def merge(self, obj_id: str) -> None:
        ...


class RasterStackBuilder(DataMerger):
    def __init__(self) -> None:
        super().__init__()

    @override
    def merge(self, obj_id: str) -> None:
        self._surfs = utils.geom.read_surfaces(obj_id)

        out_path = (
            f"{config.env('TEMP_DIR')}"
            f"{obj_id}"
            f"{config.var('RASTER_STACK')}"
            # NOTE: IRIS does not support .TIFF files.
            f"{config.var('TIF')}"
        )
        if utils.file.exists(out_path):
            return
        in_paths = [
            f"{config.env('TEMP_DIR')}{obj_id}{img_tp}{config.var('TIFF')}"
            for img_tp in [
                config.var("RGB"),
                config.var("NIR"),
                config.var("REFLECTANCE"),
                config.var("SLOPE"),
            ]
        ]
        out_meta = raster.DefaultProfile()
        # NOTE: The RGB image contains 3 bands.
        out_meta.update(count=len(in_paths) + 2)

        rgb: rasterio.io.DatasetReader
        with rasterio.open(in_paths[0]) as rgb:
            out_meta.update(width=rgb.width, height=rgb.height, transform=rgb.transform)
            rgb_data = rgb.read()

        stack: rasterio.io.DatasetWriter
        with rasterio.open(out_path, mode="w", **out_meta) as stack:
            stack.write(rgb_data, indexes=[1, 2, 3])
            for band_id, path in enumerate(in_paths[1:], start=4):
                tmp: rasterio.io.DatasetReader
                with rasterio.open(path) as tmp:
                    # Convert the units of the reflectance raster from decibels
                    # to the optical amplitude ratio between the actual and a
                    # reference target.
                    # See http://tinyurl.com/3b6jx2ax for more information.
                    # NOTE: This ensures that the output band contains only positive
                    #       values,
                    #       and thus zero-valued pixels correspond only to the
                    #       background.
                    tmp_data = tmp.read(indexes=1)
                    if band_id == 5:
                        tmp_data = 10 ** (0.1 * tmp_data)
                    stack.write(tmp_data, indexes=band_id)
