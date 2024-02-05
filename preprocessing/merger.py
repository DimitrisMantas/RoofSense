from abc import abstractmethod
from typing import Optional

import geopandas as gpd
import rasterio
import rasterio.mask
from typing_extensions import override

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
        self._surfs = utils.geom.buffer(utils.geom.read_surfaces(obj_id))

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
            rgb_data = rasterio.mask.mask(
                rgb, self._surfs[config.var("DEFAULT_GM_FIELD_NAME")]
            )[0]

        out: rasterio.io.DatasetWriter
        tmp: rasterio.io.DatasetReader
        with rasterio.open(out_path, "w", **out_meta) as out:
            out.write(rgb_data, indexes=[1, 2, 3])
            for band_id, path in enumerate(in_paths[1:], start=4):
                with rasterio.open(path) as tmp:
                    tmp_data = rasterio.mask.mask(
                        tmp,
                        self._surfs[config.var("DEFAULT_GM_FIELD_NAME")],
                        # Mask the background with zeros instead of no-data values.
                        # NOTE: This parameter does not need to be specified when
                        #       masking BM5 imagery
                        #       because their no-dara value is null,
                        #       and thus rasterio automatically replaces it with zero.
                        nodata=0,
                        # Squeeze the output data array.
                        indexes=1,
                    )[0]
                    out.write(tmp_data, indexes=band_id)
