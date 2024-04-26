import glob
import json
import os.path
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from typing_extensions import override


# TODO: Add an option to load only certain image bands.


class TrainingDataset(NonGeoDataset):
    all_bands = ["Red", "Green", "Blue", "Reflectance", "Slope"]
    rgb_bands = ["Red", "Green", "Blue"]

    classes_filename = "classes.json"
    colors_filename = "classes.json"

    classes: list[str] | None = None
    colors: ListedColormap | None = None

    valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str,
        split: str,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        # TODO: Initialize class fields lazily.
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        # Verify the dataset.
        if not self._verify():
            # TODO: Check whether the dataset is present on disk but not yet
            #  extracted and finally whether it should be downloaded.
            raise RuntimeError("Failed to verify dataset integrity.")

        # Validate the split.
        if split not in self.valid_splits:
            raise RuntimeError(
                f"Failed to verify data split: {split!r}. Allowed values are: {self.valid_splits!r}"
            )

        # Add the missing metadata.
        # TODO: Refactor this block into a separate function.
        try:
            with open(os.path.join(self.root, self.classes_filename)) as classes:
                self.classes = json.load(classes)
        except FileNotFoundError:
            raise RuntimeWarning(
                f"Failed to locate class names: {self.classes_filename!r} in dataset root folder: {self.root!r}. The file does not exist."
            )

        try:
            with open(os.path.join(self.root, self.colors_filename)) as colors:
                self.colors = ListedColormap(np.array(json.load(colors))[:, 3] / 255)
        except FileNotFoundError:
            raise RuntimeWarning(
                f"Failed to locate class names: {self.classes_filename!r} in dataset root folder: {self.root!r}. The file does not exist."
            )

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self._load(
                self.img_paths[index]
            ).float(),
            "mask": self._load(
                self.msk_paths[index]
            ).long(),
        }

    @override
    def __len__(self) -> int:
        return len(self.img_paths)

    def plot(self, sample: dict[str, Tensor]) -> Figure:
        # Extract the image and corresponding mask from the sample.
        image: np.ndarray[tuple[Any, Any, Any], np.dtype[np.uint8]] = (
            sample["image"].numpy().astype(np.uint8).squeeze()
        )
        # BxHxW -> HxWxB
        image = np.moveaxis(image, source=0, destination=-1)

        mask: np.ndarray[tuple[Any, Any], np.dtype[np.uint8]] = (
            sample["mask"].numpy().astype(np.uint8).squeeze()
        )

        # Set up the plot and extract the model prediction if it exists.
        num_cols = 2
        has_pred = "prediction" in sample
        if has_pred:
            num_cols = 3
            pred: np.ndarray[tuple[Any, Any], np.dtype[np.uint8]] = sample[
                "prediction"
            ].numpy()

        fig: Figure
        axs: np.ndarray[tuple[Literal[2, 3]], np.dtype[np.object_]]
        fig, axs = plt.subplots(
            1, ncols=num_cols, figsize=(num_cols * 4, 5), layout="constrained"
        )

        # Plot the sample.
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[0].set_title("Image")

        mask_opts = {
            "cmap": self.colors,
            "vmin": 0,
            "vmax": self.colors.N,
            "interpolation": "none",
        }

        axs[1].imshow(mask, **mask_opts)
        axs[1].axis("off")
        axs[0].set_title("User Mask")

        if has_pred:
            axs[0].imshow(pred, **mask_opts)
            axs[0].axis("off")
            axs[0].set_title("Model Mask")

        return fig

    @lru_cache
    def _load(self, filename: str) -> Tensor:
        f: rasterio.io.DatasetReader
        with rasterio.open(filename) as f:
            return torch.as_tensor(f.read())

    def _verify(self) -> bool:
        self.img_dir = os.path.join(self.root, "imgs")
        self.msk_dir = os.path.join(self.root, "msks")
        if not os.path.exists(self.img_dir) or not os.path.exists(self.msk_dir):
            return False

        self.img_paths = glob.glob(os.path.join(self.img_dir, "*.tif"))
        self.msk_paths = glob.glob(os.path.join(self.msk_dir, "*.tif"))
        if len(self.img_paths) == 0 or len(self.msk_paths) == 0:
            return False

        img_names = [
            os.path.basename(path).split(".", maxsplit=1)[0] for path in self.img_paths
        ]
        msk_names = [
            os.path.basename(path).split(".", maxsplit=1)[0] for path in self.msk_paths
        ]
        return img_names == msk_names
