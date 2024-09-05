from __future__ import annotations

import os.path
import re
from enum import IntEnum, auto
from functools import lru_cache
from typing import AnyStr, Final, TypedDict

import cjio.cityjson
import cjio.cjio
import cjio.models
import geopandas as gpd
import numpy as np
import rasterio.features
import rasterio.io
import rasterio.mask
import shapely
from geopandas import GeoDataFrame

# See https://github.com/pangeo-data/cog-best-practices for more information.
os.environ.update(
    {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
)


class LevelOfDetail(IntEnum):
    LOD_12 = 0
    LOD_13 = auto()
    LOD_22 = auto()


class MaskGeneralizer:
    TILE_ID_REGEX: Final[re.Pattern[AnyStr]] = re.compile(r"\d{1,2}-\d{3,4}-\d{2,4}")

    def __init__(self, dirpath: str) -> None:
        self._dirpath = dirpath
        self._parser = _BAG3DParser(dirpath)

    # NOTE: Use this function to generalize ground truth and prediction masks for evaluation.
    def generalize(
        self,
        src_filepath: str,
        dst_filepath: str,
        lod: LevelOfDetail,
        preserve_background: bool = True,
    ) -> None:
        """Generalize a pixel-wise segmentation mask to a given building level of detail (LoD)."""
        tile_id = self._resolve_tile_id(src_filepath)
        all_surfs = self._parser.parse(tile_id, lod)

        src: rasterio.io.DatasetReader
        with rasterio.open(src_filepath) as src:
            bbox = gpd.GeoDataFrame(
                {"id": [0], "geometry": [shapely.box(*src.bounds)]}, crs="EPSG:28992"
            )
            data = src.read(indexes=1)
            meta = src.profile

        # TODO: Check whether this operation is necessary.
        valid_surfs = all_surfs.overlay(bbox).geometry

        surf_labels = range(1, len(valid_surfs) + 1)
        mask = rasterio.features.rasterize(
            shapes=((surf, val) for surf, val in zip(valid_surfs, surf_labels)),
            out_shape=src.shape,
            transform=src.transform,
        )
        for i, label in enumerate(surf_labels):
            # print(f"{i}/{len(valid_surfs)}")
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

    def _resolve_tile_id(self, filepath: str) -> str:
        match = self.TILE_ID_REGEX.findall(filepath)
        if not match:
            raise ValueError(
                f"Filepath {filepath} does not contain valid 3DBAG tile ID."
            )
        return match[0]


class _GeopandasDict(TypedDict):
    id: list[str]
    geometry: list[shapely.Polygon]


class _BAG3DParser:
    def __init__(self, dirpath: str) -> None:
        super().__init__()
        self._dirpath = dirpath

        # NOTE: These fields are updated at the beginning of each parsing operation.
        self._data: cjio.cityjson.CityJSON | None = None
        self._surfs: _GeopandasDict | None = None

    @lru_cache
    def parse(self, tile_id: str, lod: LevelOfDetail) -> GeoDataFrame:
        self._update(tile_id)
        buildings: dict[str, cjio.models.CityObject]
        buildings = self._data.get_cityobjects(type="building")
        for building in buildings.values():
            self._parse_building_parts(building, lod)
        return gpd.GeoDataFrame(self._surfs, crs="EPSG:28992")

    def _update(self, tile_id: str) -> None:
        self._data: cjio.cityjson.CityJSON = cjio.cityjson.load(
            os.path.join(self._dirpath, f"{tile_id}.city.json")
        )
        # TODO: Check whether we can add a constructor for this dictionary.
        self._surfs: _GeopandasDict = {"id": [], "geometry": []}

    def _parse_building_parts(
        self, building: cjio.models.CityObject, lod: LevelOfDetail
    ) -> None:
        parts: dict[str, cjio.models.CityObject]
        parts = self._data.get_cityobjects(id=building.children)
        for part in parts.values():
            self._parse_surfaces(building, part, lod)

    def _parse_surfaces(
        self,
        building: cjio.models.CityObject,
        building_part: cjio.models.CityObject,
        lod: LevelOfDetail,
    ) -> None:
        part_geom: cjio.models.Geometry
        # Parse the surfaces.
        # NOTE: The LoD 1.1, 1.2, and 2.2 representations appear first, second,
        #       and third in the corresponding array, respectively.
        part_geom = building_part.geometry[lod]
        # Parse the roof surfaces.
        # NOTE: The wall and roof surfaces appear first and second in the corresponding
        #       array, respectively.
        part_surfs = part_geom.surfaces[1]
        part_surfs = part_geom.get_surface_boundaries(part_surfs)
        for surf in part_surfs:
            # Parse the exterior surface boundary.
            surf = shapely.force_2d(shapely.Polygon(surf[0]))
            # TODO: Check whether this dictionary can be populated automatically.
            self._surfs["id"].append(building.id)
            self._surfs["geometry"].append(surf)


if __name__ == "__main__":
    import config

    config.config()

    gen = MaskGeneralizer(dirpath=config.env("TEMP_DIR"))
    for name, lod in zip(
        ["lod12", "lod13", "lod22"],
        [LevelOfDetail.LOD_12, LevelOfDetail.LOD_13, LevelOfDetail.LOD_22],
    ):
        gen.generalize(
            src_filepath=r"C:\Documents\RoofSense\dataset\9-284-556.map.mask.tif",
            dst_filepath=f"9-284-556.map.mask.{name}.tif",
            lod=lod,
        )
