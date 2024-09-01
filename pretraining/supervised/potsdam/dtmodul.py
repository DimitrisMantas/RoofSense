import os
import warnings
from collections.abc import Sequence
from itertools import zip_longest
from typing import Any

import matplotlib as mpl
import matplotlib.figure
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.utils import dataset_split
from torchgeo.datasets import NonGeoDataset, Potsdam2D
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential
from torchgeo.transforms.transforms import _RandomNCrop
from typing_extensions import override

from augmentations.feature import MinMaxScaling
from pretraining.supervised.potsdam.dataset import Sample


class PotsdamRGBDataModule(NonGeoDataModule):
    def __init__(
        self,
        dataset_class: type[NonGeoDataset],
        batch_size: int = 8,
        lengths: Sequence[float] = (0.8, 0.2),
        num_workers: int | None = None,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_class, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.lengths = lengths

        max_num_workers = max(1, os.cpu_count() // 2)
        if num_workers is None:
            self.num_workers = min(batch_size, max_num_workers)
        elif batch_size < num_workers <= max_num_workers:
            msg = (
                f"Number of specified worker processes: {num_workers!r} greater than "
                f"batch size: {batch_size!r}. Improved performance due to excess "
                f"workers is not guaranteed."
            )
            warnings.warn(msg, UserWarning)
        else:
            msg = (
                f"Number of specified worker processes: {num_workers!r} greater than "
                f"half the total number of logical CPUs in the system: "
                f"{max_num_workers!r}. Windows systems may suffer from erroneous "
                f"behaviour and out-of-memory errors due to excessive memory paging. "
                f"See See https://tinyurl.com/439cb683 and "
                f"https://tinyurl.com/yc63wxey for more information."
            )
            warnings.warn(msg, ResourceWarning)

        if self.num_workers == 0 and persistent_workers:
            msg = (
                "No worker processes specified to be persisted. Value of "
                "'persistent_workers' will be set to 'False'."
            )
            warnings.warn(msg, UserWarning)
            self.persistent_workers = False
        else:
            self.persistent_workers = persistent_workers

        self.pin_memory = pin_memory

        self.aug = AugmentationSequential(
            MinMaxScaling(mins=torch.tensor([0] * 3), maxs=torch.tensor([255] * 3)),
            data_keys=["image", "mask"],
        )

    @override
    def plot(self, sample: Sample) -> mpl.figure.Figure | None:
        fig: Figure | None = None
        if hasattr(self.dataset_class, "plot"):
            fig = self.dataset_class.plot(sample)
        return fig

    def setup(self, stage: str) -> None:
        self.dataset = self.dataset_class(**self.kwargs)
        splits = torch.utils.data.random_split(
            self.dataset, self.lengths, generator=torch.Generator().manual_seed(0)
        )
        for name, split in zip_longest(["train", "val", "test"], splits):
            # if split is None:
            #     raise RuntimeError(f"No subset defined for {stage} stage.")
            setattr(self, f"{name}_dataset", split)

    @override
    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=getattr(self, f"{split}_dataset"),
            batch_size=self.batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(0),
            persistent_workers=self.persistent_workers,
        )


class Potsdam2DDataModule(NonGeoDataModule):
    def __init__(
        self,
        dataset_class: type[Potsdam2D],
        batch_size: int = 64,
        patch_size: tuple[int, int] | int = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Potsdam2DDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Potsdam2D`.
        """
        super().__init__(dataset_class, batch_size=1, num_workers=num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.aug = AugmentationSequential(
            MinMaxScaling(mins=torch.tensor([0] * 3), maxs=torch.tensor([255] * 3)),
            _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = self.dataset_class(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset = dataset_split(
                self.dataset, self.val_split_pct
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(split="test", **self.kwargs)

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=False,
        )
