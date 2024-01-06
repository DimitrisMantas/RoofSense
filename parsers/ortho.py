import geopandas as gpd
import rasterio.merge

import config
from parsers._base import DataParser
from parsers.utils import raster


# TODO: Clean up
class OrthoDataParser(DataParser):
    def __init__(self, obj_id: str) -> None:
        super().__init__(obj_id)

    def parse(self, index: gpd.GeoDataFrame) -> None:
        ###########################
        # Fetch the image names.
        # TODO: The initializer should take in the image names.
        obj_filename = (
            f"{config.env('TEMP_DIR')}"
            f"{self.obj_id}"
            f"{config.var('DEFAULT_SURFACES_FOOTPRINT_FILE_ID')}"
            f"{config.var('GEOPACKAGE')}"
        )
        obj = gpd.read_file(obj_filename)
        # TODO:Check lidar.ipynb.
        obj["geometry"] = obj["geometry"].buffer(10)

        img_ids = index.overlay(obj)["id_1"].unique()
        img_nms_cir = [
            f"{config.env('TEMP_DIR')}{'CIR'}_{id_}"
            for id_ in img_ids
            # for tp in ["RGB", "CIR"]
        ]
        img_nms_rgb = [
            f"{config.env('TEMP_DIR')}{'RGB'}_{id_}"
            for id_ in img_ids
            # for tp in ["RGB", "CIR"]
        ]
        ###########################
        # Crop the RGB
        cropped_names_rgb = []
        for nm in img_nms_rgb:
            i_nm = f"{nm}{config.var('TIFF')}"
            o_nm = f"{nm}{'.crop'}{config.var('TIFF')}"
            cropped_names_rgb.append(o_nm)
            raster.crop(i_nm, o_nm, obj.total_bounds)
        # Merge the RGB.
        rgb_merged, rgb_merged_transform = rasterio.merge.merge(cropped_names_rgb)
        with rasterio.open(
            f"{config.env('TEMP_DIR')}{self.obj_id}.rgb{config.var('TIFF')}",
            "w",
            width=rgb_merged.shape[2],
            height=rgb_merged.shape[1],
            count=3,
            transform=rgb_merged_transform,
            **raster.DefaultProfile(),
        ) as f:
            f.write(rgb_merged)
        ###########################
        # Crop the CIR
        cropped_names_cir = []
        for nm in img_nms_cir:
            i_nm = f"{nm}{config.var('TIFF')}"
            o_nm = f"{nm}{'.crop'}{config.var('TIFF')}"
            cropped_names_cir.append(o_nm)
            # NOTE: Keep the NIR band onnly.
            raster.crop(i_nm, o_nm, obj.total_bounds, bands=1)
        # Merge the CIR.
        cir_merged, cir_merged_transform = rasterio.merge.merge(cropped_names_cir)
        with rasterio.open(
            f"{config.env('TEMP_DIR')}{self.obj_id}.nir{config.var('TIFF')}",
            "w",
            width=cir_merged.shape[2],
            height=cir_merged.shape[1],
            count=1,
            transform=cir_merged_transform,
            **raster.DefaultProfile(),
        ) as f:
            f.write(cir_merged)
        ###########################
        # Merge the cropped RGB and CIR
        cropped_names = [
            f"{config.env('TEMP_DIR')}{self.obj_id}.rgb{config.var('TIFF')}",
            f"{config.env('TEMP_DIR')}{self.obj_id}.nir{config.var('TIFF')}",
        ]
        with rasterio.open(cropped_names[0]) as f:
            meta = f.meta
        meta.update(count=4)
        with rasterio.open(
            f"{config.env('TEMP_DIR')}{self.obj_id}{config.var('TIFF')}", "w", **meta
        ) as f:
            with rasterio.open(cropped_names[0]) as src1:
                f.write_band(1, src1.read(1))
                f.write_band(2, src1.read(2))
                f.write_band(3, src1.read(3))
            with rasterio.open(cropped_names[1]) as src2:
                f.write_band(4, src2.read(1))
