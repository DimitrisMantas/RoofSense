from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from roofsense.augmentations.feature import MinMaxScaling
from roofsense.pretraining.supervised.airs.dataset import AIRSDataset


class AIRSDataModule(NonGeoDataModule):
    def __init__(
        self,
        dataset_class: type[AIRSDataset],
        batch_size: int = 64,
        # patch_size: tuple[int, int] | int = 64,
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
        super().__init__(
            dataset_class, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        # self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.aug = AugmentationSequential(
            MinMaxScaling(mins=torch.tensor([0] * 3), maxs=torch.tensor([255] * 3)),
            # _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = self.dataset_class(**self.kwargs)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset,
                lengths=[1 - self.val_split_pct, self.val_split_pct],
                generator=torch.Generator().manual_seed(0),
            )

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
            pin_memory=True,
        )
