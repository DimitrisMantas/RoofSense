from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio.io
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor
from torchgeo.datasets.geo import RasterDataset
from torchgeo.datasets.utils import BoundingBox


class TrainingDataset(RasterDataset):
    # TODO: Populate these fields.
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
        "Background",
        "Other",
        "Asphalt Shingles",
        "Bituminous Membranes",
        "Clay Tiles",
        "Loose Gravel",
        "Metal",
        "Solar Panels",
        "Vegetation",
    )
    # The class-color map.
    cmap = {
        0: (255, 255, 255, 85),
        1: (128, 128, 128, 85),
        2: (141, 211, 199, 85),
        3: (255, 255, 179, 85),
        4: (190, 186, 218, 85),
        5: (251, 128, 114, 85),
        6: (128, 117, 211, 85),
        7: (253, 180, 100, 85),
        8: (179, 222, 105, 85),
    }

    def __init__(
        self,
        root: str | os.PathLike[str],
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        self.root = root
        self.download = download
        self.checksum = checksum

        lc_colors = np.zeros((max(self.cmap.keys()) + 1, 4))
        lc_colors[list(self.cmap.keys())] = list(self.cmap.values())
        lc_colors = lc_colors[:, :3] / 255
        self._lc_cmap = ListedColormap(lc_colors)

        self._verify()

        super().__init__(root, transforms=transforms, cache=cache)

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

    def __getitem__(self, query: int | str | BoundingBox) -> dict[str, Any]:
        """
        Fetch an image and its corresponding mask by dataset order,
        file name, or spatial bounds.

        # TODO: Complete the docstring of this method.
        :param query: The query index. If it is an integer, ``n``, the ``n``-th image
                      and mask are returned based on the lexicographical order of the
                      file names in the dataset. If it is a string representing an image
                      or mask file name, the corresponding pair is returned. If it is
                      a coordinate interleaved axis-aligned bounding box, the...

        :return: A dictionary containing the queried image and corresponding mask. If
                 the query is a ``BoundingBox`` instance, its spatial extents are also
                 returned.
        """
        if isinstance(query, int):
            sample = self._getitem_int(query)
        elif isinstance(query, str):
            sample = self._getitem_str(query)
        elif isinstance(query, BoundingBox):
            sample = self._getitem_box(query)
        else:
            raise ValueError(f"Invalid query {query!r} of type {type(query)!r}.")
        return sample

    def _getitem_int(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        img_matches = self.index.intersection(self.index.bounds, objects=True)
        img_path = [match.object for match in img_matches][index]
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
        """Load a single image.

        Args:
            path: unique ID of the image

        Returns:
            the image
        """
        f: rasterio.io.DatasetReader
        with rasterio.open(path) as f:
            # todo: what is the diff between torch.tensor and torch.from_numpy
            return torch.tensor(f.read())

    def _getitem_str(self, query: str) -> dict[str, Any]:
        pass

    def _getitem_box(self, query: BoundingBox) -> dict[str, CRS | BoundingBox | Tensor]:
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
            raise IndexError(  # TODO: Rephrase this error message.
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
        image = sample["image"][rgb_indices].permute(1, 2, 0)

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
