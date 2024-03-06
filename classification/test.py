from typing import Any

import kornia.augmentation as K
import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets import LandCoverAI
from torchgeo.transforms import AugmentationSequential

from classification.augmentations import MinMaxScaling
from classification.datamodules import TrainingDataModule
from classification.task import TrainingTask


class LandCoverAIDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the LandCover.ai dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LandCoverAIDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.LandCoverAI`.
        """
        super().__init__(LandCoverAI, batch_size, num_workers, **kwargs)

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.mins = torch.tensor([0, 0, 0])
        self.maxs = torch.tensor([255, 255, 255])
        self.train_aug = AugmentationSequential(
            MinMaxScaling(mins=self.mins, maxs=self.maxs),
            # K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            # K.RandomSharpness(p=0.5),
            # K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=["image", "mask"],
        )
        self.aug = AugmentationSequential(
            MinMaxScaling(mins=self.mins, maxs=self.maxs), data_keys=["image", "mask"]
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
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    # datamodule=LandCoverAIDataModule(root="data/LandCoverAI",batch_size=8,num_workers=8,persistent_workers=True,pin_memory=True)
    datamodule = TrainingDataModule(
        root="data/LandCoverAI",
        batch_size=8,
        patch_size=512,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="logs/LandCoverAI", save_top_k=10, save_last=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=50
    )

    batch_size_callback = lightning.pytorch.callbacks.BatchSizeFinder()

    logger = TensorBoardLogger(save_dir="logs/LandCoverAI")
    task = TrainingTask(
        model="unet",
        backbone="resnet18",
        weights=True,
        in_channels=3,
        num_classes=5,
        loss="ce",
    )

    profiler = AdvancedProfiler(dirpath="logs/LandCoverAI/profiling")
    # todo check stategies + callbacks + profiler
    torch.set_float32_matmul_precision("high")
    trainer = Trainer(
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            # lightning.pytorch.callbacks.BatchSizeFinder(),
            # lightning.pytorch.callbacks.LearningRateFinder(),
            # lightning.pytorch.callbacks.GradientAccumulationScheduler(scheduling={0: 5}),
            lightning.pytorch.callbacks.LearningRateMonitor(),
        ],
        log_every_n_steps=1,
        logger=logger,
        benchmark=True,
        # profiler=profiler,
        # max_epochs=5
        # detect_anomaly=True
        # fast_dev_run=True
    )
    trainer.fit(model=task, datamodule=datamodule)
