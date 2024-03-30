from __future__ import annotations

import os
import pathlib
import re
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any, Callable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.io
import rasterio.merge
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from torch import Tensor
from torchgeo.datasets.geo import (GeoDataset,
                                   UnionDataset,
                                   IntersectionDataset,
                                   RasterDataset, )
from torchgeo.datasets.utils import BoundingBox, disambiguate_timestamp


class HybridDataset(GeoDataset, ABC):
    # NOTE: The dataset is assumed to have no geospatial functionality until its
    # spatial index has been populated.
    # The coordinate reference system of the dataset, as specified by the first file
    # read to populate its spatial index.
    # This file is determined by the lexicographic order of the files in the dataset.
    _crs: Optional[CRS] = None
    # The spatial resolution of the dataset, as specified by the first file read to
    # populate its spatial index.
    # This file is determined by the lexicographic order of the files in the dataset.
    _res: Optional[float] = None

    def __init__(
        self,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        use_index: bool = False,
    ) -> None:
        # Cache the file list of the dataset.
        # NOTE: Caching the output of corresponding property directly using a
        #       relevant function decorator does not allow for instances of this class
        #       to be garbage collected.
        # NOTE: Use a NumPy array to store the file list of the dataset to avoid "memory
        #       leaks"
        #       when sampling it
        #       using data loaders with a relatively large total number of worker
        #       threads.
        #
        #       See https://tinyurl.com/439cb683 and https://tinyurl.com/yc63wxey for
        #       more information.
        self._files: np.ndarray[tuple[Any,], np.dtype[np.string_]] = np.asarray(
            super().files
        ).astype(np.string_)
        # NOTE: This avoids initializing any potential sibling classes in the case of
        # multiple inheritance.
        GeoDataset.__init__(self, transforms)
        if use_index:
            self.populate_index()

    @property
    def bounds(self) -> BoundingBox:
        if len(self.index):
            # NOTE: The corresponding parent method requires the spatial index of the
            # dataset to be populated.
            return GeoDataset.bounds.fget(self)
        else:
            msg = (
                "Unable to query the spatial index of the dataset. Populate the "
                "spatial index of the dataset to enable is geospatial functionality."
            )
            warnings.warn(msg, UserWarning)

    @GeoDataset.crs.setter
    def crs(self, crs: CRS) -> None:
        if len(self.index):
            # NOTE: The corresponding parent method requires the spatial index of the
            # dataset to be populated.
            GeoDataset.crs.fset(self, crs)
        else:
            msg = (
                "Unable to mutate the CRS of the dataset. Populate the spatial index "
                "of the dataset to enable its geospatial functionality."
            )
            warnings.warn(msg, UserWarning)

    @property
    def files(self) -> np.ndarray[tuple[Any,], np.dtype[np.string_]]:
        return self._files

    @abstractmethod
    def populate_index(self) -> None:
        ...

    @abstractmethod
    def _getitem_int(self, query: int) -> dict[str, Any]:
        ...

    @abstractmethod
    def _getitem_box(self, query: BoundingBox) -> dict[str, Any]:
        ...

    def __and__(self, other: GeoDataset) -> IntersectionDataset:
        self.populate_index()
        return IntersectionDataset(self, other)

    def __getitem__(self, query: int | BoundingBox) -> dict[str, Any]:
        if isinstance(query, int):
            item = self._getitem_int(query)
        elif isinstance(query, BoundingBox):
            self.populate_index()
            item = self._getitem_box(query)
        else:
            msg = f"Encountered invalid query {query!r} of type {type(query)!r}."
            raise ValueError(msg)
        return item

    def __getstate__(
        self,
    ) -> tuple[dict[str, Any], list[tuple[Any, Any, Optional[Any]] | None]]:
        if len(self.index):
            # NOTE: The corresponding parent method requires the spatial index of the
            # dataset to be populated.
            state = super().__getstate__()
        else:
            state = self.__dict__, []
        return state

    def __len__(self) -> int:
        if len(self.index):
            len_ = len(self.index)
        else:
            len_ = len(self.files)
        return len_

    def __or__(self, other: GeoDataset) -> UnionDataset:
        self.populate_index()
        return UnionDataset(self, other)

    def __str__(self) -> str:
        is_index_populated = int(bool(len(self.index)))
        rpr = (
            f"{self.__class__.__name__}"
            f"\n\t"
            f"Length: {len(self)}"
            f"\n\t"
            f"Geospatial Functionality: {['Disabled', 'Enabled'][is_index_populated]}"
        )
        return rpr


class HybridRasterDataset(HybridDataset, RasterDataset, ABC):
    def __init__(
        self,
        paths: str | Iterable[str],
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        use_index: bool = False,
    ) -> None:
        self.paths = paths
        self.bands = bands or self.all_bands
        self.cache = cache
        if not self.separate_files:
            self.band_indexes = None
            if self.bands:
                if self.all_bands:
                    self.band_indexes = [
                        self.all_bands.index(i) + 1 for i in self.bands
                    ]
                else:
                    msg = (
                        "Unable to query the raster bands of the dataset.  Specify "
                        "all channels of the dataset in addition to the ones to "
                        "the ones to return."
                    )
                    raise ValueError(msg)
        # NOTE: This avoids initializing any potential sibling classes in the case of
        # multiple inheritance.
        HybridDataset.__init__(self, transforms=transforms, use_index=use_index)

    def populate_index(self) -> None:
        if len(self.index):
            return
        else:
            msg = (
                "The spatial index of the dataset is being populated. This operation "
                "may require a considerable amount of time to complete."
            )
            warnings.warn(msg, UserWarning)
            self._populate_index()

    # TODO: Review this method.
    def _populate_index(self) -> None:
        i = 0
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in self.files:
            filepath = str(filepath, encoding="utf-8")
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass
                        if self._crs is None:
                            crs = src.crs
                        if self._res is None:
                            res = src.res[0]
                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index._insert(i)
                    i += 1
        if i == 0:
            msg = (
                f"No {self.__class__.__name__} data was found "
                f"in `paths={self.paths!r}'`"
            )
            if self.bands:
                msg += f" with `bands={self.bands}`"
            raise FileNotFoundError(msg)
        self._crs = cast(CRS, crs)
        self._res = cast(float, res)


# TODO: Review this class.
class TrainingDataset(HybridRasterDataset):
    # The URL to the training dataset used in the RoofSense publication.
    download_path = ""
    # The file name of the corresponding archive.
    download_name = ""
    # The MD5 hash of the archive.
    download_hash = ""

    # The path pattern to the training images.
    filename_glob = os.path.join("imgs", "*.tif")
    filename_regex = ".*tif"

    # The names of the bands present in the training images.
    all_bands = ("Red", "Green", "Blue", "Near-infrared", "Reflectance", "Slope")
    rgb_bands = ("Red", "Green", "Blue")

    # The names of the classes present in the training masks.
    classes = (
        "Asphalt Shingles",
        "Bituminous Membranes",
        "Clay Tiles",
        "Loose Gravel",
        "Metal",
        "Solar Panels",
        "Vegetation",
        "Other",
        "Invalid",
        "__ignore__",
    )
    # The class-color map.
    cmap = {
        0: (141, 211, 199, 85),
        1: (255, 255, 179, 85),
        2: (190, 186, 218, 85),
        3: (251, 128, 114, 85),
        4: (128, 117, 211, 85),
        5: (253, 180, 100, 85),
        6: (179, 222, 105, 85),
        7: (252, 205, 229, 85),
        8: (128, 128, 128, 85),
        9: (255, 255, 255, 85),
    }

    def __init__(
        self,
        root: str | os.PathLike[str],
        download: bool = False,
        checksum: bool = False,
        bands: Optional[Sequence[str]] = None,
        cache: bool = True,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        use_index: bool = False,
    ) -> None:
        self.root = root
        self.download = download
        self.checksum = checksum

        lc_colors = np.zeros((max(self.cmap.keys()) + 1, 4))
        lc_colors[list(self.cmap.keys())] = list(self.cmap.values())
        lc_colors = lc_colors[:, :3] / 255
        self._lc_cmap = ListedColormap(lc_colors)

        self._verify()
        super().__init__(
            root, bands=bands, cache=cache, transforms=transforms, use_index=use_index
        )

    def _verify(self):
        """Ensure that the dataset is valid."""
        if self._verify_data():
            return

        # Check if the zip file has already been downloaded
        pathname = os.path.join(self.root, self.download_name)
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _verify_data(self) -> bool:
        """Verify if the images and masks are present."""
        # 1. Check image and mask folder names.
        img_dir = pathlib.Path(self.root).joinpath("imgs")
        msk_dir = pathlib.Path(self.root).joinpath("msks")
        if not img_dir.exists() or not msk_dir.exists():
            return False
        # 2. Check image and mask counts.
        img_paths = sorted(img_dir.glob("*.tif"))
        msk_paths = sorted(msk_dir.glob("*.tif"))
        if not len(img_paths) or not len(msk_paths):
            return False
        # 3. Check image and mask names.
        img_names = [path.stem for path in img_paths]
        msk_names = [path.stem for path in msk_paths]
        return img_names == msk_names

    def _download(self) -> None:
        """Download the dataset."""
        # download_url(self.url, self.root, md5=self.md5 if self.checksum else None)
        pass

    def _extract(self) -> None:
        """Extract the dataset."""
        # extract_archive(os.path.join(self.root, self.filename))
        pass

    def _getitem_int(self, index: int) -> dict[str, Tensor]:
        if len(self.index):
            match = list(self.index.intersection(self.index.bounds, objects=True))[
                index
            ]
            return self._getitem_box(BoundingBox(*match.bounds))

        img_path = str(self.files[index], encoding="utf-8")
        msk_path = img_path.replace("imgs", "msks")

        sample = {
            "image": self._load_image(img_path).float(),
            "mask": self._load_image(msk_path).long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @lru_cache
    def _load_image(self, path: str) -> Tensor:
        f: rasterio.io.DatasetReader
        with rasterio.open(path) as f:
            return torch.as_tensor(f.read())

    def _getitem_box(self, query: BoundingBox) -> dict[str, CRS | BoundingBox | Tensor]:
        self.populate_index()
        # Find the images intersecting the bounding box.
        # NOTE: The index is an rtree.RTree instance containing the spatial bounds and
        #       name of each image in the dataset as a single object.
        img_matches = self.index.intersection(
            tuple(query),
            # Ensure that the image names are returned.
            objects=True,
        )
        img_paths = [match.object for match in img_matches]
        if not img_paths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        img = self._merge_files(img_paths, query, self.band_indexes)

        # Fetch the corresponding masks.
        msk_paths = [path.replace("imgs", "msks") for path in img_paths]
        # noinspection PyTypeChecker
        msk = self._merge_files(msk_paths, query)

        # NOTE: The image and mask sample fields are mandatory and should be named as
        #       such.
        sample = {
            "crs": self.crs,
            "bbox": query,
            # NOTE: Images are represented as 32-bit floating-point tensors.
            "image": img.float(),
            # NOTE: Masks are represented as 32-bit fixed-point tensors.
            "mask": msk.long(),
        }
        # Apply the user-defined transforms.
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")
        image = sample["image"].squeeze()[rgb_indices].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())
        mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy().astype("uint8").squeeze()
            num_panels += 1

        kwargs = {
            "cmap": self._lc_cmap,
            "vmin": 0,
            "vmax": len(self.cmap),
            "interpolation": "none",
        }
        fig, axs = plt.subplots(ncols=num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, **kwargs)
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            # noinspection PyUnboundLocalVariable
            axs[2].imshow(predictions, **kwargs)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
