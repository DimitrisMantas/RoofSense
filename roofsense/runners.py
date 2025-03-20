import os
import warnings
from collections.abc import Iterable

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from roofsense.training.task import TrainingTask


def train_supervised(
    task: TrainingTask,
    datamodule,
    log_dirpath: str,
    study_name: str | None = None,
    experiment_name: int | str | None = None,
    ckpt_dirname: str | None = "ckpts",
    ckpt_filename: str | None = "best",
    callbacks: Callback | Iterable[Callback] | None = None,
    test: bool = True,
    **kwargs,
) -> Trainer:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(0, workers=True)

    logger = TensorBoardLogger(
        save_dir=log_dirpath, name=study_name, version=experiment_name
    )
    model_ckpt = ModelCheckpoint(
        dirpath=logger.log_dir
        if ckpt_dirname is None
        else os.path.join(logger.log_dir, ckpt_dirname),
        filename=ckpt_filename,
        monitor=task.monitor_optim,
        mode="max",
        save_last=True,
    )
    # Match log and checkpoint version numbers in the case of automatic versioning.
    model_ckpt.STARTING_VERSION = 0

    cbs = [LearningRateMonitor(), model_ckpt, RichProgressBar()]
    if callbacks is not None:
        callbacks = [callbacks] if isinstance(callbacks, Callback) else callbacks
        for cb in callbacks:
            cbs.append(cb)
    trainer = Trainer(logger=logger, callbacks=cbs, benchmark=True, **kwargs)

    with warnings.catch_warnings(action="ignore", category=UserWarning):
        trainer.fit(model=task, datamodule=datamodule)
        if test:
            trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

    return trainer
