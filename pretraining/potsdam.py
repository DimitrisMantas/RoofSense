import os
import warnings
from collections.abc import Callable
from typing import Any

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint,
                                         RichProgressBar, )
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.utils import dataset_split
from torchgeo.datasets import Potsdam2D
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential
from torchgeo.transforms.transforms import _RandomNCrop

from augmentations.feature import MinMaxScaling
from training.loss import DistribBasedLoss, RegionBasedLoss
from training.task import TrainingTask


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
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.files = []
        for name in self.splits[split]:
            image = os.path.join(root, self.image_root, name) + "_RGB.tif"
            mask = os.path.join(root, name) + "_label.tif"
            if os.path.exists(image) and os.path.exists(mask):
                self.files.append(dict(image=image, mask=mask))


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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(42, workers=True)

    datamodule = Potsdam2DDataModule(
        dataset_class=Potsdam2DRBG,
        root=r"C:\Users\Dimit\Downloads\Potsdam",
        batch_size=2,
        patch_size=512,
        num_workers=2,
    )

    task = TrainingTask(
        decoder="deeplabv3plus",
        encoder="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=6,
        loss_params={
            "this": DistribBasedLoss.CROSS,
            "this_kwargs": {"label_smoothing": 0.05},
            "that": RegionBasedLoss.DICE,
            "that_kwargs": {"squared_pred": True},
            "ignore_background": True,
        },
    )

    logger = TensorBoardLogger(
        save_dir="../logs", name="pretraining", version="potsdam-rgb"
    )

    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "ckpts"),
        filename="best",
        monitor="val/loss",
        save_last=True,
    )
    # Match log and checkpoint version numbers in the case of automatic versioning.
    model_ckpt.STARTING_VERSION = 0

    trainer = Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), model_ckpt, RichProgressBar()],
        max_epochs=100,
        benchmark=True,
    )

    with warnings.catch_warnings(action="ignore", category=UserWarning):
        trainer.fit(model=task, datamodule=datamodule)
