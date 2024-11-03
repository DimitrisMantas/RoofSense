import gzip
import hashlib
import os
import re
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, auto, verify
from typing import AnyStr, Final

import geopandas as gpd
import requests

from roofsense.utils.file import confirm_write_op, get_default_data_dir


@dataclass
class BAG3DTileAssetInfo:
    tid: list[str]
    url: list[str]


@dataclass
class BAG3DTileAssetManifest:
    img: BAG3DTileAssetInfo
    lidr: BAG3DTileAssetInfo

    def save(self, filename: str, overwrite: bool = False) -> None:
        confirm_write_op(filename, type="file", overwrite=overwrite)
        raise NotImplementedError


@verify(UNIQUE)
class LevelOfDetail(StrEnum):
    LoD12 = auto()
    LoD13 = auto()
    LoD22 = auto()


class BAG3DTileStore:
    _BASE_URL: Final[str] = "https://data.3dbag.nl/"
    # TODO: Narrow this down
    _TILE_FMT: Final[re.Pattern[AnyStr]] = re.compile(r"^\d{1,2}-\d{3,4}-\d{2,4}$")

    def __init__(self, version: str = "2024.02.28") -> None:
        self._dir = get_default_data_dir(version)
        self._ver = version
        self._init_version()

    @property
    def index(self) -> gpd.GeoDataFrame:
        return self._index

    @property
    def version(self) -> str:
        return self._ver

    def download_index(self, overwrite: bool = False, **kwargs):
        filename = os.path.join(self._dir, "tile_index.fgb")
        confirm_write_op(filename, type="file", overwrite=overwrite)
        self._index.to_file(filename, **kwargs)

    # TODO: Make downloads optional if possible.
    def download_tile(
        self, tile_id: str, checksum: bool = True, overwrite: bool = False, **kwargs
    ) -> None:
        valid_fmt = self._validate_tile_id(tile_id)

        filename = os.path.join(self._dir, f"{tile_id}.gpkg")
        confirm_write_op(filename, type="file", overwrite=overwrite)

        # TODO: Check that this works.
        match = self._index.loc[self._index.tile_id == valid_fmt]

        url: str = match["gpkg_download"].item()
        if checksum:
            true_sha: str = match["gpkg_sha256"].item()
            curr_sha = hashlib.sha256()
        with requests.get(url=url, **kwargs) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                # https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
                delete=False,
            ) as temp:
                for chunk in r.iter_content(
                    # Write the file as it arrives.
                    chunk_size=None
                ):
                    temp.write(chunk)
                    if checksum:
                        curr_sha.update(chunk)
        if checksum:
            if curr_sha.hexdigest() != true_sha:
                raise RuntimeError(
                    f"Failed to verify the integrity of tile: {tile_id}. Try downloading again with the checksum option disabled."
                )
        with gzip.open(temp.name, mode="rb") as src, open(filename, mode="wb") as dst:
            shutil.copyfileobj(src, dst)
        os.unlink(temp.name)

    def read_tile(self, tile_id: str, lod: LevelOfDetail) -> gpd.GeoDataFrame:
        filename = os.path.join(self._dir, f"{tile_id}.gpkg")
        return gpd.read_file(filename, layer=f"{lod}_2d", force_2d=True)

    def sample_tile(self) -> gpd.GeoSeries:
        raise NotImplementedError

    def asset_manifest(
        self, tile_id: str, image_index: gpd.GeoDataFrame, lidar_index: gpd.GeoDataFrame
    ) -> BAG3DTileAssetManifest:
        surfs = self.read_tile(tile_id, lod=LevelOfDetail.LoD22)

        # TODO: Do not rebuild the manifest if it already exists.

        image_matches = image_index.overlay(surfs)
        lidar_matches = lidar_index.overlay(surfs)

        return BAG3DTileAssetManifest(
            img=BAG3DTileAssetInfo(
                tid=image_matches.tid.unique(), url=image_matches.url.unique()
            ),
            lidr=BAG3DTileAssetInfo(
                tid=lidar_matches.tid.unique(), url=lidar_matches.url.unique()
            ),
        )

    def _init_version(self) -> None:
        url = urllib.parse.urljoin(
            self._BASE_URL, f"v{self._ver.replace('.', '')}/tile_index.fgb"
        )
        self._index = gpd.read_file(url)

    def _validate_tile_id(self, tile_id: str) -> str:
        valid_fmt = tile_id.replace("-", "/")
        if not self._TILE_FMT.fullmatch(tile_id):
            raise ValueError(
                f"Invalid tile ID: {tile_id}. Make sure to delimit digit groups with dashes."
            )
        if valid_fmt not in self._index.tile_id.values:
            raise ValueError(f"Failed to locate tile with ID: {tile_id} in index.")
        return valid_fmt


if __name__ == "__main__":
    BAG3DTileStore().download_tile("9-284-556")
    BAG3DTileStore().asset_manifest(
        "9-284-556", image_index=gpd.GeoDataFrame(), lidar_index=gpd.GeoDataFrame()
    ).save(
        # TODO: Resolve the filename automatically.
    )
    temp = 1
