import warnings

import lightning.pytorch
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (EarlyStopping,
                                         LearningRateMonitor,
                                         ModelCheckpoint, )
from lightning.pytorch.loggers import TensorBoardLogger

from training.datamodule import TrainingDataModule
from training.loss import DistribBasedLoss, RegionBasedLoss
from training.task import TrainingTask

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(42, workers=True)

    task = TrainingTask(
        # Decoder Configuration
        model="unet",
        # Encoder Configuration
        backbone="resnet18",
        weights=True,
        # I/O Layer Configuration
        in_channels=5,
        num_classes=8 + 1,
        # Loss Configuration
        loss_params={
            "this": DistribBasedLoss.CROSS,
            "that": RegionBasedLoss.JACC,
            "ignore_background": True,
            "weight": torch.tensor(
                np.load("../dataset/temp/weights.npy"), dtype=torch.float32
            ),
        },
    )

    datamodule = TrainingDataModule(
        root="../dataset/temp", batch_size=16, num_workers=8
    )

    # todo check strategies + callbacks + profiler
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir="../logs/RoofSense"),
        callbacks=[
            ModelCheckpoint(
                dirpath="../logs/RoofSense",
                filename="best",
                monitor="val/loss",
                save_last=True,
            ),
            EarlyStopping(monitor="val/loss", patience=1000),
            LearningRateMonitor(),
        ],
        log_every_n_steps=1,
        benchmark=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning
        )
        trainer.fit(
            model=task,
            datamodule=datamodule
        )
        trainer.test(model=task, datamodule=datamodule)
