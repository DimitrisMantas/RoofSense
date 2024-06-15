import os
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
        # Architecture Configuration
        # NOTE: We choose to not experiment with any purely transformer-based
        # architectures due to their limited effectiveness with small datasets such
        # as ours.
        # Instead, we prefer to exploit the inherent inductive biases of CNNs to
        # maintain reasonable performance.
        encoder="resnet50",
        # Decoder Configuration
        decoder="deeplabv3plus",
        model_params={
            # Use the custom model implementation.
            "custom": True,
            # Add a squeeze-and-excitation block after each atrous convolution and
            # the feature concatenation stage in the ASPP module.
            "attention": False,
            # Use larger feature maps to better parse small objects.
            "encoder_output_stride": 16
        },
        # Loss Configuration
        loss_params={
            "this": DistribBasedLoss.CROSS,
            "this_kwargs": {
                # Account for potential annotation errors.
                "label_smoothing": 0.05
            },
            "that": RegionBasedLoss.DICE,
            "ignore_background": True,
            "weight": torch.from_numpy(np.load("../dataset/temp/weights.npy")).to(
                torch.float32
            ),
        },
    )

    datamodule = TrainingDataModule(root="../dataset/temp")

    logger = TensorBoardLogger(save_dir="../logs", name="training", version="base-customModel")

    model_ckpt = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "ckpts"),
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
            # TODO: Check if this hinders training.
            lightning.pytorch.callbacks.EarlyStopping(
                monitor="val/loss",
                verbose=True,
                # NOTE: We employ high early stopping patience in the exploration
                # phase to ensure that configurations which are relatively slow train
                # but still performant overall are not discarded accidentally.
                # Our patience is reduced to 50 epochs in the exploitation stage to
                # prune potentially over-optimistic learning rate schedules and thus
                # promote stable training.
                patience=100,
            ),
            lightning.pytorch.callbacks.LearningRateMonitor(),
            model_ckpt,
            lightning.pytorch.callbacks.RichProgressBar(),
            # TODO: Check if this hinders training.
            # FIXME: This doesn't work with LearningRateFinder.
            lightning.pytorch.callbacks.SpikeDetection(
                warmup=task.hparams.warmup_epochs,
                exclude_batches_path=os.path.join(logger.log_dir, "spike"),
                finite_only=False,
            ),
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
