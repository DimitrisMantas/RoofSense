from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.parse
from collections.abc import Iterable
from enum import UNIQUE, StrEnum, auto, verify
from typing import AnyStr, Final, Self

import geopandas as gpd
import numpy as np
import requests
import shapely

from roofsense.utilities.file import (
    ThreadedFileDownloader,
    confirm_write_op,
    get_default_dirpath,
)


@dataclasses.dataclass(frozen=True, slots=True)
class TileAssetInfo:
    tid: list[str]
    url: list[str]


@dataclasses.dataclass(frozen=True, slots=True)
class TileAssetManifest:
    image: TileAssetInfo
    lidar: TileAssetInfo
    tid: str

    def __post_init__(self) -> None:
        for name in ["image", "lidar"]:
            attr = getattr(self, name)
            if not isinstance(attr, TileAssetInfo):
                object.__setattr__(self, name, TileAssetInfo(**attr))

    @classmethod
    def read(cls, path: str) -> TileAssetManifest:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def downl(self, dirpath: str, overwrite: bool | Iterable[bool] = False) -> Self:
        urls = self.image.url + self.lidar.url
        pths = [os.path.join(dirpath, os.path.basename(url)) for url in urls]
        if not isinstance(overwrite, Iterable):
            overwrite = [overwrite] * len(pths)
        for i, (pth, url) in enumerate(zip(pths.copy(), urls.copy())):
            if not confirm_write_op(pth, overwrite[i]):
                pths.remove(pth)
                urls.remove(url)
        with requests.Session() as s:
            ThreadedFileDownloader(urls, filenames=pths, session=s).download()
        return self

    def save(self, dirpath: str, overwrite: bool = False) -> Self:
        path = os.path.join(dirpath, f"{self.tid}.info.json")
        if not confirm_write_op(path, overwrite=overwrite):
            return
        with open(path, mode="w") as f:
            json.dump(dataclasses.asdict(self), f)
        return self


@verify(UNIQUE)
class LevelOfDetail(StrEnum):
    LoD12 = auto()
    LoD13 = auto()
    LoD22 = auto()


class BAG3DTileStore:
    _BASE_URL: Final[str] = "https://data.3dbag.nl/"
    # TODO: Narrow this down.
    _TILE_FMT: Final[re.Pattern[AnyStr]] = re.compile(r"\d{1,2}-\d{3,4}-\d{2,4}")

    def __init__(self, dirpath: str | None = None, version: str = "2024.02.28") -> None:
        self._dirpath = get_default_dirpath(version) if dirpath is None else dirpath
        self._random = np.random.default_rng(42)
        self._init_version(version)

    @property
    def dirpath(self) -> str:
        return self._dirpath

    @property
    def index(self) -> gpd.GeoDataFrame:
        return self._index

    @property
    def random(self) -> np.random.Generator:
        return self._random

    @property
    def version(self) -> str:
        return self._ver

    def asset_manifest(
        self,
        tile_id: str,
        image_index: gpd.GeoDataFrame | None = None,
        lidar_index: gpd.GeoDataFrame | None = None,
    ) -> TileAssetManifest:
        tile_id = self._validate_tile_id(tile_id)
        surfs = self.read_tile(tile_id, lod=LevelOfDetail.LoD22)

        path = os.path.join(self.dirpath, f"{tile_id}.info.json")
        if os.path.exists(path):
            return TileAssetManifest.read(path)
        elif image_index is None or lidar_index is None:
            msg = f"Failed to locate the asset manifest corresponding to tile {tile_id!r} in the local filesystem ({path!r}). To regenerate it, please provide the required image and lidar tile indices."
            raise ValueError(msg)

        image_matches = image_index.overlay(surfs)
        lidar_matches = lidar_index.overlay(surfs)

        return TileAssetManifest(
            image=TileAssetInfo(
                tid=image_matches.tid.unique().tolist(),
                url=image_matches.url.unique().tolist(),
            ),
            lidar=TileAssetInfo(
                tid=lidar_matches.tid.unique().tolist(),
                url=lidar_matches.url.unique().tolist(),
            ),
            tid=tile_id,
        )

    # TODO: Make downloads optional if possible.
    def download_tile(
        self, tile_id: str, checksum: bool = True, overwrite: bool = False, **kwargs
    ) -> None:
        tile_id = self._validate_tile_id(tile_id)

        path = os.path.join(self._dirpath, f"{tile_id}.gpkg")
        if not confirm_write_op(path, overwrite=overwrite):
            return

        match = self.index.loc[self.index["tile_id"] == tile_id]

        url: str = match["gpkg_download"].item()
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
            true_sha256: str = match["gpkg_sha256"].item()
            curr_sha256 = hashlib.sha256()
            with open(temp.name, mode="rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    curr_sha256.update(chunk)
            if curr_sha256.hexdigest() != true_sha256:
                os.unlink(temp.name)
                raise RuntimeError(
                    f"Failed to verify the integrity of the data pack corresponding tile: {tile_id}. To force the relevant download operation, please disable the 'checksum' option."
                )

        with gzip.open(temp.name, mode="rb") as src, open(path, mode="wb") as dst:
            shutil.copyfileobj(src, dst)
        os.unlink(temp.name)

    def read_tile(self, tile_id: str, lod: LevelOfDetail) -> gpd.GeoDataFrame:
        tile_id = self._validate_tile_id(tile_id)
        path = os.path.join(self._dirpath, f"{tile_id}.gpkg")
        if not os.path.exists(path):
            self.download_tile(tile_id)
        return gpd.read_file(path, layer=f"{lod}_2d", force_2d=True).to_crs(
            crs="EPSG:28992"
        )

    def sample_tile(self, seeds: gpd.GeoDataFrame, radius: float = 15e3) -> list[str]:
        seed = seeds.sample(random_state=self.random)["geometry"]
        smp = gpd.GeoDataFrame(
            {"id": [0], "geometry": [self._sample_point_around_seed(seed, radius)]},
            crs="EPSG:28992",
        )
        return self.index.overlay(smp, keep_geom_type=False)["tile_id"]

    def _init_version(self, version: str) -> None:
        self._ver = version
        # TODO: Check if index has already been downloaded before opening from URL.
        url = urllib.parse.urljoin(
            self._BASE_URL, f"v{self.version.replace('.', '')}/tile_index.fgb"
        )
        self._index = gpd.read_file(url).to_crs("EPSG:28992")
        self._index["tile_id"] = self.index["tile_id"].str.replace("/", "-")

    def _sample_point_around_seed(
        self, seed: shapely.Point, radius: float
    ) -> shapely.Point:
        off = self.random.multivariate_normal(
            mean=[0, 0],
            # NOTE: The specified covariance along the x- and y-axes ensures that
            #       approximately 99.7% of the sampled points are located within the
            #       specified distance from their corresponding seed.
            cov=[[1 / 9, 0], [0, 1 / 9]],
        )
        nrm = np.linalg.norm(off)
        if nrm > 1:
            off /= np.linalg.norm(off)
        off *= radius
        return shapely.Point(seed.x + off[0], seed.y + off[1])

    def _validate_tile_id(self, tile_id: str) -> str:
        delimiters = [
            tile_id[match.start()] for match in re.finditer(r"[^0-9]", tile_id)
        ]
        if len(set(delimiters)) != 1:
            msg = f"The tile ID: {tile_id} is delimited inconsistently. Please use only dashes to delimit digit groups."
            raise ValueError(msg)

        delimiter = delimiters[0]
        if delimiter != "-":
            msg = f"The tile ID: {tile_id} is not delimited dashes."
            raise ValueError(msg)

        if not self._TILE_FMT.fullmatch(tile_id):
            msg = f"The tile ID: {tile_id} is invalid."
            raise ValueError(msg)

        if tile_id not in self.index["tile_id"].values:
            raise ValueError(f"Failed to locate tile with ID: {tile_id} in index.")
        return tile_id.replace("/", "-")

    def resolve_tile_id(self, filepath: str) -> str:
        # TODO: Find the most appropriate search method.
        return self._TILE_FMT.search(filepath).group(0)


if __name__ == "__main__":
    store = BAG3DTileStore()
    store.download_tile(
        "9-284-556"
        # overwrite=True
    )
    manf = store.asset_manifest(
        "9-284-556",
        image_index=gpd.read_file(r"/roofsense/data/index/image/image.gpkg"),
        lidar_index=gpd.read_file(r"/roofsense/data/index/lidar/lidar.gpkg"),
    )
    manf.save(dirpath=store.dirpath)
    manf.download(store.dirpath, overwrite=True)
    sample = store.sample_tile(gpd.read_file(r"/roofsense/data/cities.gpkg"))
    print(sample)
