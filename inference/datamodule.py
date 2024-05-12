import os
import warnings
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule
from torchgeo.samplers import BatchGeoSampler, GridGeoSampler
from torchgeo.transforms import AugmentationSequential
from typing_extensions import override

from common.augmentations import MinMaxScaling
from inference.dataset import InferenceDataset


class InferenceDataModule(GeoDataModule):
    # TODO: Share these values with TrainingDataModule.
    mins = torch.tensor([0, 0, 0, 0, 0])
    maxs = torch.tensor([255, 255, 255, 1, 90])

    def __init__(
        self,
        batch_size: int = 16,
        patch_size: int = 512,
        stride: int | None = None,
        num_workers: int | None = None,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            InferenceDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.stride = stride if stride is not None else self.patch_size // 2

        # TODO: Share this block with TrainingDataModule.
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
            MinMaxScaling(self.mins, self.maxs), data_keys=["image"]
        )

    @override
    def setup(self, stage: str) -> None:
        if stage != "predict":
            raise ValueError("Only inference tasks are supported.")

        self.predict_dataset = InferenceDataset(**self.kwargs)
        self.predict_sampler = GridGeoSampler(
            self.predict_dataset, size=self.patch_size, stride=self.stride
        )

    # TODO: Clean up this method.
    @override
    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
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
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(0),
            persistent_workers=self.persistent_workers,
        )
