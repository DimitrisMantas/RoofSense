import geopandas as gpd
import numpy as np
import rasterio.features
import rasterio.io
import rasterio.mask
import shapely
from tqdm import tqdm

from roofsense.bag3d import BAG3DTileStore, LevelOfDetail


class MaskGeneralizer:
    def __init__(self, tile_store: BAG3DTileStore) -> None:
        self._tile_store = tile_store

    # NOTE: Use this function to generalize ground truth and prediction masks for evaluation.
    def generalize(
        self,
        src_filepath: str,
        dst_filepath: str,
        lod: LevelOfDetail,
        preserve_background: bool = True,
    ) -> None:
        """Generalize a pixel-wise segmentation mask to a given building level of detail (LoD)."""
        tile_id = self._tile_store.resolve_tile_id(src_filepath)
        all_surfs = self._tile_store.read_tile(tile_id, lod)

        src: rasterio.io.DatasetReader
        with rasterio.open(src_filepath) as src:
            bbox = gpd.GeoDataFrame(
                {"id": [0], "geometry": [shapely.box(*src.bounds)]}, crs="EPSG:28992"
            )
            data = src.read(indexes=1)
            meta = src.profile

        # This speeds up the process when partial tiles are processed.
        valid_surfs = all_surfs.overlay(bbox).geometry

        surf_labels = range(1, len(valid_surfs) + 1)
        mask = rasterio.features.rasterize(
            shapes=((surf, val) for surf, val in zip(valid_surfs, surf_labels)),
            out_shape=src.shape,
            transform=src.transform,
        )
        for label in tqdm(surf_labels, desc=self.__class__.__name__):
            query = mask == label
            if preserve_background:
                query = np.logical_and(query, data != 0)
            match = data[query]
            values, counts = np.unique(match, return_counts=True)
            if np.size(counts) == 0:
                # The whole region is labelled as background.
                continue
            data[query] = values[np.argmax(counts)]

        dst: rasterio.io.DatasetWriter
        with rasterio.open(dst_filepath, mode="w", **meta) as dst:
            dst.write(data, indexes=1)
