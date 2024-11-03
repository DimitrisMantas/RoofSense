import glob
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Final, Required

import matplotlib as mpl
import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, Potsdam2D
from typing_extensions import NotRequired, TypedDict, override


class Sample(TypedDict):
    image: Required[Tensor]
    mask: Required[Tensor]
    prediction: NotRequired[Tensor]


class PotsdamRGBDataset(NonGeoDataset):
    image_dirname: Final[str] = "images"
    mask_dirname: Final[str] = "masks"
    file_glob = "*.tif"
    classes: Final[list[str]] = Potsdam2D.classes
    colors: Final[mpl.colors.ListedColormap] = mpl.colors.ListedColormap(
        np.asarray(Potsdam2D.colormap) / 255
    )

    def __init__(
        self,
        root: str,
        cache: bool = True,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    ) -> None:
        self.root = root
        self.cache = cache
        self.transforms = transforms

        self.image_dirpath = os.path.join(root, self.image_dirname)
        self.image_paths = glob.glob(os.path.join(self.image_dirpath, self.file_glob))
        self.mask_paths = [
            path.replace(self.image_dirname, self.mask_dirname)
            for path in self.image_paths
        ]

        if len(self) == 0:
            raise ValueError("Dataset is empty.")

    @override
    def __getitem__(self, index: int) -> Sample:
        image = self._load(self.image_paths[index])
        mask = self._load(self.mask_paths[index])
        sample: Sample = {
            "image": image.to(torch.float32),
            "mask": mask.to(torch.int64),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @override
    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def plot(sample: Sample) -> Figure:
        image = sample["image"].numpy().squeeze()
        image = np.moveaxis(image, source=0, destination=-1)

        mask = sample["mask"].numpy().squeeze().astype(np.uint8)

        num_cols = 2
        has_pred = "prediction" in sample
        if has_pred:
            num_cols = 3
            pred = sample["prediction"].numpy()

        fig: Figure
        axs: Sequence[Axes]
        fig, axs = plt.subplots(nrows=1, ncols=num_cols, layout="constrained")

        # Plot the sample.
        axs[0].imshow(image, interpolation="bilinear")
        axs[0].axis("off")
        axs[0].set_title("Image")

        mask_opts = {
            "cmap": PotsdamRGBDataset.colors,
            "vmin": 0 - 0.5,
            "vmax": (len(PotsdamRGBDataset.classes) - 1) + 0.5,
            "interpolation": "nearest",
        }

        temp = axs[1].imshow(mask, **mask_opts)
        axs[1].axis("off")
        axs[1].set_title("Ground Truth")

        if has_pred:
            axs[2].imshow(pred, **mask_opts)
            axs[2].axis("off")
            axs[2].set_title("Prediction")

        plt.colorbar(
            temp,
            ax=axs[2] if has_pred else axs[1],
            ticks=np.arange(0, len(PotsdamRGBDataset.classes) + 1),
        )

        return fig

    def _load(self, filepath: str) -> Tensor:
        if self.cache:
            return self._load_cache(filepath)
        else:
            return self._load_impl(filepath)

    @lru_cache
    def _load_cache(self, filepath: str) -> Tensor:
        return self._load_impl(filepath)

    def _load_impl(self, filepath: str) -> Tensor:
        src: rasterio.io.DatasetReader
        with rasterio.open(filepath) as src:
            return torch.from_numpy(src.read())


class Potsdam2DRBG(Potsdam2D):
    filenames = ["2_Ortho_RGB.zip", "5_Labels_all.zip"]
    image_root = "2_Ortho_RGB"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        checksum: bool = False,
    ) -> None:
        super().__init__(root, split, transforms, checksum)

        assert split in self.splits
        self.root = root
        self.split = split
        self.checksum = checksum

        self._verify()

        self.files = []
        for name in self.splits[split]:
            image = os.path.join(root, self.image_root, name) + "_RGB.tif"
            mask = os.path.join(root, name) + "_label.tif"
            if os.path.exists(image) and os.path.exists(mask):
                self.files.append(dict(image=image, mask=mask))
