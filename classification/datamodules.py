from __future__ import annotations

import os
import warnings
from typing import Optional, Any

# noinspection PyPep8Naming
import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import (GeoDataModule,
                                  MisconfigurationException,
                                  BaseDataModule, )
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler, BatchGeoSampler
from torchgeo.transforms import AugmentationSequential

from classification import splits
from classification.augmentations import MinMaxScaling
from classification.datasets import TrainingDataset


class TrainingDataModule(GeoDataModule):
    # The minimum cell value of each raster stack layer.
    mins = torch.tensor([0, 0, 0, 0, 0, 0])

    # The maximum cell value of each raster stack layer.
    maxs = torch.tensor([255, 255, 255, 255, 1, 90])

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: int | tuple[int, int] = 512,
        # The total number of steps constituting the length of each training epoch.
        length: Optional[int] = None,
        num_workers: Optional[int] = None,
        persistent_workers: bool = True,
        pin_memory: bool = True,
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
        max_workers = max(os.cpu_count() // 2, 1)
        if num_workers is None:
            self.num_workers = max_workers
        elif num_workers > num_workers:
            msg = (
                "The total number of  data loader worker threads may be too large. "
                "This can result in in abnormally large amount of virtual memory "
                "being registered, which can in-turn lead to erroneous behavior and "
                "out-of-memory errors."
                + "\n"
                + "Monitor memory consumption during the model fitting stage and "
                "consider decreasing the total number of  data loader worker "
                "threads to at most half the count of logical processors on your "
                "machine."
                + "\n"
                + "See https://tinyurl.com/439cb683 and https://tinyurl.com/yc63wxey "
                "for  more information."
            )
            warnings.warn(msg, UserWarning)

        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        # General Augmentations
        self.aug = AugmentationSequential(
            MinMaxScaling(self.mins, self.maxs), data_keys=["image", "mask"]
        )

        # Training Augmentations
        # NOTE: This field overwrites the predefined augmentations.
        self.train_aug = AugmentationSequential(  # Scaling
            MinMaxScaling(self.mins, self.maxs),  # Geometric Augmentations
            # Flips
            K.RandomVerticalFlip(),
            K.RandomHorizontalFlip(),  # TODO: Rotations
            # TODO: Intensity Augmentations
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        dataset = TrainingDataset(**self.kwargs)
        dataset.populate_index()

        generator = torch.Generator().manual_seed(0)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = splits.random_grid_split(
            dataset, lengths=[0.7, 0.15, 0.15], size=8, generator=generator
        )

        if self.patch_size >= 512:
            # The dataset will be operated in non-geospatial mode, and thus no
            # sampler is necessary.
            if self.patch_size > 512:
                warnings.warn(
                    f"The requested patch size is larger than {512} px. Will "
                    f"perform online learning with individual tiles as samples."
                )
        else:
            if stage == "fit":
                # TODO: Check how many times the index is populated during training;
                #  it should be once per split (i.e., three times).
                self.train_dataset.populate_index()
                self.train_batch_sampler = RandomBatchGeoSampler(
                    self.train_dataset, self.patch_size, self.batch_size, self.length
                )
            if stage in ["fit", "validate"]:
                self.val_dataset.populate_index()
                self.val_sampler = GridGeoSampler(
                    self.val_dataset, self.patch_size, self.patch_size
                )
            if stage == "test":
                self.test_dataset.populate_index()
                self.test_sampler = GridGeoSampler(
                    self.test_dataset, self.patch_size, self.patch_size
                )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        try:
            sampler = self._valid_attribute(
                f"{split}_batch_sampler", f"{split}_sampler", "batch_sampler", "sampler"
            )
            if isinstance(sampler, BatchGeoSampler):
                batch_size = 1
                batch_sampler, sampler = sampler, None
            else:
                batch_sampler = None
            shuffle = None
        except MisconfigurationException:
            # The dataset will be operated in non-geospatial mode.
            batch_sampler = sampler = None
            shuffle = split == "train"

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def transfer_batch_to_device(
        self, batch: dict[str, Tensor], device: torch.device, dataloader_idx: int
    ) -> dict[str, Tensor]:
        # NOTE: Samples do not have these attributes when the corresponding dataset
        # is operated in non-geospatial mode.
        batch.pop("crs", None)
        batch.pop("bbox", None)

        batch = BaseDataModule.transfer_batch_to_device(
            self, batch, device, dataloader_idx
        )
        return batch
