import os
import warnings

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from roofsense.pretraining.supervised.airs.dataset import AIRSDataset
from roofsense.pretraining.supervised.airs.dtmodul import AIRSDataModule
from roofsense.training.task import TrainingTask

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(42, workers=True)

    datamodule = AIRSDataModule(
        dataset_class=AIRSDataset,
        root_path=r"C:\Users\Dimit\Downloads\AIRS",
        batch_size=8,
        # patch_size=512,
        num_workers=8,
    )

    task = TrainingTask(
        decoder="deeplabv3plus",
        encoder="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=2,
        loss_params={
            "names": [
                "crossentropyloss"
                # , "diceloss"
            ]
            # "weight": torch.from_numpy(np.fromfile("../dataset/temp/weights.bin")).to(
            #     torch.float32
            # ),
            # "label_smoothing": 0.05,
            # "squared_pred": True,
        },
    )

    logger = TensorBoardLogger(
        save_dir="../../../../logs",
        name="pretraining",
        version="airs_batch-size-2_max-epochs-100",
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
