from __future__ import annotations

import json
import os.path
import warnings
from collections.abc import Callable, Sequence
from enum import CONTINUOUS, UNIQUE, IntEnum, auto, verify
from functools import lru_cache
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from torch import Tensor, classproperty
from torchgeo.datasets import NonGeoDataset
from typing_extensions import override


@verify(CONTINUOUS, UNIQUE)
class Band(IntEnum):
    RED = auto()
    GRN = auto()
    BLU = auto()
    RFL = auto()
    SLP = auto()

    @classproperty
    def ALL(cls) -> list[Band]:
        return list(cls)

    @classproperty
    def RGB(cls) -> list[Band]:
        return list(cls)[:3]


# TODO: Plot single bands.


class TrainingDataset(NonGeoDataset):
    classes_filename = "names.json"
    colors_filename = "colors.json"
    splits_filename = "splits.json"

    classes: list[str] | None = None
    colors: ListedColormap | None = None

    def __init__(
        self,
        root: str,
        split: Literal["training", "validation", "test"],
        download: bool = False,
        checksum: bool = False,
        bands: Band | list[Band] = Band.ALL,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    ) -> None:
        # TODO: Initialize class fields lazily.
        self.root = root
        self.split = split
        self.download = download
        self.checksum = checksum

        self.bands = bands if isinstance(bands, list) else [bands]
        # Verify the band list.
        self.has_plt = len(self.bands) == 1 or len(self.bands) >= 3
        self.has_rgb = False
        if len(self.bands) >= 3:
            for i, band in enumerate(Band.RGB):
                if band != self.bands[i]:
                    warnings.warn(
                        f"Failed to locate RGB bands in specified band list: {self.bands!r}. One or more bands missing or out of order.",
                        RuntimeWarning,
                    )
                    break
            self.has_rgb = True

        self.transforms = transforms

        # Verify the dataset.
        self.img_paths: list[str] | None = None
        self.msk_paths: list[str] | None = None
        with open(os.path.join(self.root, self.splits_filename)) as splits:
            names = json.load(splits)[split]
        if not self._verify(names):
            # TODO: Check whether the dataset is present on disk but not yet
            #  extracted and finally whether it should be downloaded.
            raise RuntimeError("Failed to verify dataset integrity.")

        # Add the missing metadata.
        # TODO: Refactor this block into a separate function.
        try:
            with open(os.path.join(self.root, self.classes_filename)) as classes:
                self.classes = list(json.load(classes).values())
        except FileNotFoundError:
            warnings.warn(
                f"Failed to locate class names: {self.classes_filename!r} in dataset root folder: {self.root!r}. The file does not exist.",
                RuntimeWarning,
            )

        try:
            with open(os.path.join(self.root, self.colors_filename)) as colors:
                self.colors = ListedColormap(
                    np.array(list(json.load(colors).values())) / 255
                )
        except FileNotFoundError:
            warnings.warn(
                f"Failed to locate class names: {self.classes_filename!r} in dataset root folder: {self.root!r}. The file does not exist.",
                RuntimeWarning,
            )

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = {  # "name":torch.as_tensor(
            #     [ord(c) for c in os.path.basename(self.img_paths[index]).split(".", maxsplit=1)[0]],
            #     dtype=torch.int64
            # ),
            "image": self._load_image(self.img_paths[index]),
            "mask": self._load_mask(self.msk_paths[index]),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @override
    def __len__(self) -> int:
        return len(self.img_paths)

    def plot(self, sample: dict[str, Tensor]) -> Figure:
        if not self.has_plt:
            raise NotImplementedError
            warnings.warn(  # TODO: Inform the user that images will not be plotted.
                "", RuntimeWarning
            )

        # Extract the image and corresponding mask from the sample.
        image: np.ndarray[
            tuple[Any, Any, Any], np.dtype[np.uint8] | np.dtype[np.float32]
        ] = sample["image"].numpy().squeeze()
        # BxHxW -> HxWxB
        image = np.moveaxis(image, source=0, destination=-1)

        if self.has_rgb:
            if np.amax(image) > 1:
                dtype = np.uint8
            else:
                dtype = np.float32
            slice = 3
        elif len(self.bands) == 1:
            dtype = np.float32
            slice = 1
        else:
            # TODO: Do not plot the image.
            raise NotImplementedError
        image = image.astype(dtype)[..., :slice]

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
        # axs: np.ndarray[tuple[Literal[2, 3]], np.dtype[np.object_]]
        axs: Sequence[Axes]
        fig, axs = plt.subplots(
            1,
            ncols=num_cols,  # figsize=(num_cols * 4, 5),
            layout="constrained",
        )

        # Plot the sample.
        axs[0].imshow(
            image, cmap=None if self.has_rgb else "turbo", interpolation="bilinear"
        )
        axs[0].axis("off")
        axs[0].set_title(
            f"Training Image\n({'RGB' if self.has_rgb else self.bands[0].name})"
        )

        mask_opts = {
            "cmap": self.colors,
            "vmin": 0 - 0.5,
            "vmax": (len(self.classes) - 1) + 0.5,
            "interpolation": "nearest",
        }

        temp = axs[1].imshow(mask, **mask_opts)
        axs[1].axis("off")
        axs[1].set_title("Training Mask")

        if has_pred:
            axs[2].imshow(pred, **mask_opts)
            axs[2].axis("off")
            axs[2].set_title("Prediction Mask")

        plt.colorbar(
            temp,
            ax=axs[2] if has_pred else axs[1],
            ticks=np.arange(0, len(self.classes) + 1),
        )

        # fig.suptitle(f"Chip {''.join([chr(o) for o in sample['name']])}")

        return fig

    @lru_cache
    def _load_image(self, filename: str) -> Tensor:
        src: rasterio.io.DatasetReader
        with rasterio.open(filename) as src:
            return torch.from_numpy(src.read(self.bands)).to(torch.float32)

    @lru_cache
    def _load_mask(self, filename: str) -> Tensor:
        src: rasterio.io.DatasetReader
        with rasterio.open(filename) as src:
            return torch.from_numpy(src.read()).to(torch.int64)

    def _verify(self, names) -> bool:
        # The image and mask directories exist.
        img_dir = os.path.join(self.root, "imgs")
        msk_dir = os.path.join(self.root, "msks")
        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            return False

        # The image and mask paths exist.
        self.img_paths = [os.path.join(img_dir, name) for name in names]
        self.msk_paths = [os.path.join(msk_dir, name) for name in names]
        if len(self.img_paths) == 0 or len(self.msk_paths) == 0:
            return False
        for path in self.img_paths:
            if not os.path.isfile(path):
                return False
        for path in self.msk_paths:
            if not os.path.isfile(path):
                return False

        # The image and mask names are the same.
        img_names = [
            os.path.basename(path).split(".", maxsplit=1)[0] for path in self.img_paths
        ]
        msk_names = [
            os.path.basename(path).split(".", maxsplit=1)[0] for path in self.msk_paths
        ]
        return img_names == msk_names
