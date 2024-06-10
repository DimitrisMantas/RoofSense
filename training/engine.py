import warnings

import lightning.pytorch
import numpy as np
import torch
from lightning import Trainer
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

    datamodule = TrainingDataModule(root="../dataset/temp")

    logger=TensorBoardLogger(save_dir="../logs", name="encoders", version="base")

    model_ckpt = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best",
        monitor="val/loss",
        save_last=True,
    )
    # Match log and checkpoint version numbers in the case of automatic versioning.
    model_ckpt.STARTING_VERSION = 0

    trainer = Trainer(
        logger=logger,
        # NOTE: We do not employ any gradient accumulation schemes due to the
        # existence of batch normalization layers in our encoders of choice.
        # This is because the parameters of these layers are not accumulated and
        # are instead updated only in the backpropagation step.
        # In fact, this is also why we choose not to freeze any encoder
        # parameters in the transfer learning process.
        callbacks=[
            lightning.pytorch.callbacks.EarlyStopping(
                monitor="val/loss",
                # NOTE: We employ high early stopping patience in the exploration
                # phase to ensure that configurations which are relatively slow train
                # but still performant overall are not discarded accidentally.
                # Our patience is reduced to 50 epochs in the exploitation stage to
                # prune potentially hyper-optimistic learning rate schedules and thus
                # promote stable training.
                patience=100,
            ),
            model_ckpt,
            lightning.pytorch.callbacks.RichProgressBar(),
            LearningRateMonitor(),
            lightning.pytorch.callbacks.LearningRateMonitor(),
        ],
        # NOTE: We initially train all models for 1000 epochs to investigate the full
        # convergence behavior of each configuration.
        # Once we have limited our search space to a certain architecture and a
        # particular collection of support models to sweep, we limit the training
        # duration to the maximum number of epochs required to achieve quasi-minimum
        # validation loss with similar configurations in the earlier stages of this
        # process.
        max_epochs=600,
        benchmark=True,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        trainer.fit(model=task, datamodule=datamodule)
        trainer.test(model=task, datamodule=datamodule, ckpt_path="best")
