import os
import warnings

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint,
                                         RichProgressBar, )
from lightning.pytorch.loggers import TensorBoardLogger

from training.task import TrainingTask


def train_supervised(
    task: TrainingTask,
    datamodule,
    log_dirpath: str,
    study_name: str | None = None,
    experiment_name: int | str | None = None,
    ckpt_dirname: str | None = "ckpts",
    ckpt_filename: str | None = "best",
    test: bool = True,
    **kwargs,
) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(42, workers=True)

    logger = TensorBoardLogger(
        save_dir=log_dirpath, name=study_name, version=experiment_name
    )
    model_ckpt = ModelCheckpoint(
        dirpath=logger.log_dir
        if ckpt_dirname is None
        else os.path.join(logger.log_dir, ckpt_dirname),
        filename=ckpt_filename,
        monitor=task.monitor,
        save_last=True,
    )
    # Match log and checkpoint version numbers in the case of automatic versioning.
    model_ckpt.STARTING_VERSION = 0

    trainer = Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), model_ckpt, RichProgressBar()],
        benchmark=True,
        **kwargs,
    )

    with warnings.catch_warnings(action="ignore", category=UserWarning):
        trainer.fit(model=task, datamodule=datamodule)
        if test:
            trainer.test(model=task, datamodule=datamodule, ckpt_path="best")


def train_unsupervised(task, dataloader) -> None:
    """Train a supervised model.

    Args:
        task:
        datamodule:

    Returns:

    """
    raise NotImplementedError
