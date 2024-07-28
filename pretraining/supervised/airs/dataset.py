import glob
import os
import warnings
from typing import Literal

import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset


class AIRSDataset(NonGeoDataset):
    image_dirname = "image/chips"
    image_glob = "*.tif"

    mask_dirname = "label/chips"
    mask_glob = "*.tif"

    # splits = {"train", "val", "test"}

    def __init__(self, root_path: str,
                 # split: str
                 ) -> None:
        self._root_path = root_path
        # if split not in self.splits:
        #     msg = f"Expected split to be one of {self.splits!r}, but got {split}."
        #     raise ValueError(msg)
        # self._split = split

        self._images = glob.glob(
            os.path.join(
                self.root_path,
                # self.split,
                self.image_dirname, self.image_glob
            )
        )
        self._images.sort()

        self._masks = glob.glob(
            os.path.join(self.root_path,
                         # self.split,
                         self.mask_dirname, self.mask_glob)
        )
        self._masks.sort()

        # # Remove erroneous tiles.
        # bad_indices = set()
        # for index, name in enumerate(self.images):
        #     with warnings.catch_warnings(
        #         action="ignore", category=rasterio.errors.NotGeoreferencedWarning
        #     ):
        #         src: rasterio.io.DatasetReader
        #         with rasterio.open(name) as src:
        #             if src.width != src.height:
        #                 bad_indices.add(index)
        # self._images = [
        #     self.images[i] for i in range(len(self.images)) if i not in bad_indices
        # ]
        # self._masks = [
        #     self.masks[i] for i in range(len(self.masks)) if i not in bad_indices
        # ]

    @property
    def root_path(self) -> str:
        return self._root_path

    # @property
    # def split(self) -> str:
    #     return self._split

    @property
    def images(self) -> list[str]:
        return self._images

    @property
    def masks(self) -> list[str]:
        return self._masks

    def __getitem__(self, item: int) -> dict[Literal["image", "mask"], Tensor]:
        return {
            "image": self._load(item, what="image"),
            "mask": self._load(item, what="mask"),
        }

    def __len__(self) -> int:
        return len(self.images)

    def _load(self, item: int, what: Literal["image", "mask"]) -> Tensor:
        collection = self.images if what == "image" else self.masks

        with warnings.catch_warnings(
            action="ignore", category=rasterio.errors.NotGeoreferencedWarning
        ):
            src: rasterio.io.DatasetReader
            with rasterio.open(collection[item]) as src:
                array = src.read()
        # if array.shape[-1]!=512 or array.shape[-2]!=512:
        #     print(self.images[item],self.masks[item])
        #     raise RuntimeError
        tensor = torch.from_numpy(array)
        if what == "image":
            return tensor.to(torch.float32)
        else:
            return tensor.to(torch.int64)


if __name__ == "__main__":
    dataset = AIRSDataset(root_path=r"C:\Users\Dimit\Downloads\AIRS",
                          # split="train"
                          )
    print(len(dataset.images))
    print(len(dataset.masks))
