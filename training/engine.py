import warnings

import lightning.pytorch
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
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
        model="deeplabv3+",
        # Encoder Configuration
        backbone="resnet50",
        weights=True,
        # I/O Layer Configuration
        in_channels=5,
        num_classes=8 + 1,
        # Loss Configuration
        loss_params={
            "this": DistribBasedLoss.CROSS,
            "this_kwargs":{
                "label_smoothing":0.05
            },
            "that": RegionBasedLoss.DICE,
            "ignore_background": True,
            "weight": torch.from_numpy(
                np.load("../dataset/temp/weights.npy")
            ).to(torch.float32),
        },
    )

    datamodule = TrainingDataModule(root="../dataset/temp",
                                    # # NOTE: The training dataset is too small for
                                    # # asynchronous batch loading to be beneficial.
                                    # # See https://lightning.ai/docs/pytorch/stable
                                    # # /advanced/speed.html#dataloaders for more
                                    # # information.
                                    # num_workers=0,
                                    # # NOTE: Batch loading is performed on the main
                                    # # thread, so there are no workers to persist.
                                    # persistent_workers=False
                                    )

    model_ckpt = lightning.pytorch.callbacks.ModelCheckpoint(
                dirpath="../logs/RoofSense",
                filename="best",
                monitor="val/loss",
                save_last=True,
            )
    # Match log and checkpoint version numbers in the case of automatic versioning.
    model_ckpt.STARTING_VERSION=0

    trainer = Trainer(
        logger=TensorBoardLogger(save_dir="../logs/RoofSense"),
        callbacks=[
            lightning.pytorch.callbacks.EarlyStopping(
                monitor="val/loss", patience=1000
            ),
            # lightning.pytorch.callbacks.DeviceStatsMonitor(cpu_stats=True),
            model_ckpt,
            # lightning.pytorch.callbacks.OnExceptionCheckpoint(
            #     dirpath="../logs/RoofSense"
            # ),
            lightning.pytorch.callbacks.RichProgressBar(),
            LearningRateMonitor(),
        ],
        benchmark=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        trainer.fit(model=task, datamodule=datamodule)
        trainer.test(model=task, datamodule=datamodule, ckpt_path="best")
