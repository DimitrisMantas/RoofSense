# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai dataset."""
import abc
import glob
import hashlib
import os
from functools import lru_cache
from typing import Any, Callable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor
from torch.utils.data import Dataset
from torchgeo.datasets import NonGeoDataset, RasterDataset
from torchgeo.datasets.utils import (BoundingBox,
                                     download_url,
                                     extract_archive,
                                     working_dir, )


class EarthSurfaceWaterBase(Dataset[dict[str, Any]], abc.ABC):
    url = "https://zenodo.org/records/5205674/files/dset-s2.zip?download=1"
    filename = "dset-s2.zip"
    md5 = "3268c89070e8734b4e91d531c0617e03"

    classes = ["Water", "Other"]
    cmap = {0: "#000000", 1: "#FFFFFF"}

    def __init__(
        self, root: str = "data", download: bool = False, checksum: bool = False
    ) -> None:
        self.root = root
        self.download = download
        self.checksum = checksum

        lc_colors = np.zeros((max(self.cmap.keys()) + 1, 4))
        lc_colors[list(self.cmap.keys())] = list(self.cmap.values())
        lc_colors = lc_colors[:, :3] / 255
        self._lc_cmap = ListedColormap(lc_colors)

        self._verify()

    @abc.abstractmethod
    def __getitem__(self, query: Any) -> dict[str, Any]:
        ...

    def _verify(self) -> None:
        if self._verify_dir():
            return
        # if self._verify_zip():
        #     self._extract()
        #     return
        # if not self.download:
        #     raise RuntimeError(
        #         f"Dataset not found in `root={self.root}` and `download=False`, either specify a different `root` directory or use `download=True` to automatically download the dataset."
        #     )
        self._download()
        self._extract()

    def _verify_dir(self) -> bool:
        img_query = os.path.join(self.root, "imgs", "*.tif")
        msk_query = os.path.join(self.root, "msks", "*.tif")
        imgs = glob.glob(img_query)
        msks = glob.glob(msk_query)
        return len(imgs) > 0 and len(imgs) == len(msks)

    def _verify_zip(self) -> bool:
        # TODO: Replace this block with own method.
        pathname = os.path.join(self.root, self.filename)
        return os.path.exists(pathname)

    def _download(self) -> None:
        # TODO: Replace this block with own method.
        download_url(self.url, self.root, md5=self.md5 if self.checksum else None)

    def _extract(self) -> None:
        # TODO: Replace this block with own method.
        extract_archive(os.path.join(self.root, self.filename))

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy().astype("uint8").squeeze(), 0, 3)
        mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=4, cmap=self._lc_cmap, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(
                predictions, vmin=0, vmax=4, cmap=self._lc_cmap, interpolation="none"
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class EarthSurfaceWaterGeo(EarthSurfaceWaterBase, RasterDataset):
    filename_glob = os.path.join("images", "*.tif")
    filename_regex = ".*tif"

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        EarthSurfaceWaterBase.__init__(self, root, download, checksum)
        RasterDataset.__init__(self, root, crs, res, transforms=transforms, cache=cache)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        img_filepaths = cast(list[str], [hit.object for hit in hits])
        mask_filepaths = [path.replace("images", "masks") for path in img_filepaths]

        if not img_filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        img = self._merge_files(img_filepaths, query, self.band_indexes)
        mask = self._merge_files(mask_filepaths, query, self.band_indexes)
        sample = {
            "crs": self.crs,
            "bbox": query,
            "image": img.float(),
            "mask": mask.long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class LandCoverAI(EarthSurfaceWaterBase, NonGeoDataset):
    """LandCover.ai dataset.

    See the abstract LandCoverAIBase class to find out more.

    .. note::

       This dataset requires the following additional library to be installed:

       * `opencv-python <https://pypi.org/project/opencv-python/>`_ to generate
         the train/val/test split
    """

    sha256 = "15ee4ca9e3fd187957addfa8f0d74ac31bc928a966f76926e11b3c33ea76daa1"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LandCover.ai dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in ["train", "val", "test"]

        super().__init__(root, download, checksum)

        self.transforms = transforms
        self.split = split
        with open(os.path.join(self.root, split + ".txt")) as f:
            self.ids = f.readlines()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = self.ids[index].rstrip()
        sample = {"image": self._load_image(id_), "mask": self._load_target(id_)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    @lru_cache
    def _load_image(self, id_: str) -> Tensor:
        """Load a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the image
        """
        filename = os.path.join(self.root, "output", id_ + ".jpg")
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            tensor = torch.from_numpy(array).float()
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    @lru_cache
    def _load_target(self, id_: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the target mask
        """
        filename = os.path.join(self.root, "output", id_ + "_m.png")
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            tensor = torch.from_numpy(array).long()
            return tensor

    def _verify_dir(self) -> bool:
        """Verify if the images and masks are present."""
        img_query = os.path.join(self.root, "output", "*_*.jpg")
        mask_query = os.path.join(self.root, "output", "*_*_m.png")
        images = glob.glob(img_query)
        masks = glob.glob(mask_query)
        return len(images) > 0 and len(images) == len(masks)

    def _extract(self) -> None:
        """Extract the dataset.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        super()._extract()

        # Generate train/val/test splits
        # Always check the sha256 of this file before executing
        # to avoid malicious code injection
        with working_dir(self.root):
            with open("split.py") as f:
                split = f.read().encode("utf-8")
                assert hashlib.sha256(split).hexdigest() == self.sha256
                exec(split)
