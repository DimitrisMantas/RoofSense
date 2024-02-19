from __future__ import annotations

import warnings
from typing import Optional, Any

import kornia.augmentation as K
import torch
import torchgeo.datamodules
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import random_grid_cell_assignment
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler, BatchGeoSampler
from torchgeo.transforms import AugmentationSequential

from classification.augmentations import MinMaxNormalization
from classification.datasets import TrainingDataset


class TrainingDataModule(GeoDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        patch_size: int | tuple[int, int] = 64,
        length: Optional[int] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            TrainingDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )
        self.persistent_workers = persistent_workers

        # General Augmentations
        self.aug = AugmentationSequential(
            MinMaxNormalization(), data_keys=["image", "mask"]
        )
        # Training Augmentations
        # NOTE: This field overwrites the predefined augmentations.
        self.train_aug = AugmentationSequential(  # Normalization
            MinMaxNormalization(),  # Geometric Augmentations
            # Flips
            K.RandomVerticalFlip(),
            K.RandomHorizontalFlip(),  # Rotations
            # TODO: Add rotational augmentations.
            # Intensity Augmentations
            # TODO: Add photometric augmentations.
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = TrainingDataset(**self.kwargs)

        # NOTE: This method produces the same splits per program execution!
        #       This is because the underlying spatial index returns results in no
        #       specific order.
        generator = torch.Generator().manual_seed(0)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_bbox_assignment(dataset, [0.7, 0.15, 0.15], generator)

        if stage in ["fit"]:
            # TODO
            if self.patch_size >= 512:
                if self.patch_size > 512:
                    warnings.warn(
                        f"The requested patch size is larger than {512} px. Will perform online learning with individual tiles as samples."
                    )
                self.train_sampler = torchgeo.samplers.PreChippedGeoSampler(
                    self.train_dataset,
                    # TODO: Find out why the training data not shuffled by default.
                    shuffle=True,
                )
            else:
                self.train_batch_sampler = RandomBatchGeoSampler(
                    self.train_dataset, self.patch_size, self.batch_size, self.length
                )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        sampler = self._valid_attribute(
            f"{split}_batch_sampler", f"{split}_sampler", "batch_sampler", "sampler"
        )
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        if isinstance(sampler, BatchGeoSampler):
            batch_size = 1
            batch_sampler = sampler
            sampler = None
        else:
            batch_sampler = None

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    datamodule = TrainingDataModule(root="../pretraining", patch_size=512)
    datamodule.setup("fit")
    print(len(datamodule.train_batch_sampler))
    print(datamodule.train_batch_sampler.length)
    for i, sample in enumerate(datamodule.train_batch_sampler):
        tmp = datamodule.train_dataset[sample[0]]

        # ignore empty masks
        if not torch.any(tmp["mask"]):
            continue

        datamodule.plot(tmp).show()
