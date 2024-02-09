from __future__ import annotations

from typing import Optional, Any, Dict

import kornia.augmentation as K
import torch
import torchgeo.datamodules
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datasets import random_bbox_assignment
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler, BatchGeoSampler
from torchgeo.transforms import AugmentationSequential

from classification.datasets import TrainingDataset


class MinMaxNormalize(K.IntensityAugmentationBase2D):
    """Normalize channels to the range [0, 1] using min/max values."""

    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1)
        self.flags = {"mins": mins.view(1, -1, 1, 1), "maxs": maxs.view(1, -1, 1, 1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # noinspection PyTypeChecker
        return (input - flags["mins"]) / (flags["maxs"] - flags["mins"] + 1e-10)


class TrainingDataModule(torchgeo.datamodules.GeoDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        patch_size: int | tuple[int, int] = 64,
        length: Optional[int] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        **kwargs: Any,
    ):
        """Initialize a new L7IrishDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.L7Irish`.
        """
        super().__init__(
            TrainingDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )
        self.persistent_workers = persistent_workers

        # Augmentations
        # Disable stage-agnostic augmentations.

        # self.train_aug = AugmentationSequential(
        #     # Normalize each band independently since they represent different stuff.
        #     # MinMaxNormalize,
        #     K.RandomRotation(p=0.5, degrees=90),
        #     K.RandomVerticalFlip(p=0.5),
        #     K.RandomHorizontalFlip(p=0.5),
        #     # Mess with the color of only the RGB bands
        #     # K.RandomSharpness(p=0.5),
        #     # K.ColorJiggle(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #     # This needs to be the last augmentation.
        #     # NOTE: Figure out why
        #     _RandomNCrop(_to_tuple(self.patch_size), self.batch_size),
        #     data_keys=["image", "mask"],
        # )

        # Disable data augmentation.
        self.train_aug = AugmentationSequential(
            K.RandomVerticalFlip(p=0), data_keys=["image", "mask"]
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
        ) = random_bbox_assignment(dataset, [3, 1, 1], generator)

        if stage in ["fit"]:
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
    datamodule = TrainingDataModule(root="../pretraining", patch_size=128)
    datamodule.setup("fit")
    datamodule.train_dataset.plot(datamodule.test_dataset[0]).show()
