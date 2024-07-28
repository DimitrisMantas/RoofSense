import numpy as np
import torch

from runners import train_supervised
from training.datamodule import TrainingDataModule
from training.dataset import Band
from training.task import TrainingTask

if __name__ == "__main__":
    datamodule = TrainingDataModule(
        root="../dataset/temp",
        # Do not use the density band.
        bands=Band.ALL[:-1],
        append_hsv=True,
    )

    task = TrainingTask(
        # Architecture Configuration
        # NOTE: We choose to not experiment with any purely transformer-based
        # architectures due to their limited effectiveness with small datasets such
        # as ours.
        # Instead, we prefer to exploit the inherent inductive biases of CNNs to
        # maintain reasonable predictive performance.
        encoder="resnet18",
        encoder_weights=r"C:\Documents\RoofSense\logs\pretraining\potsdam-rgbir\ckpts\best.ckpt",
        # Decoder Configuration
        decoder="deeplabv3plus",
        model_params={
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
            "weight": torch.from_numpy(np.fromfile("../dataset/temp/weights.bin")).to(
                torch.float32
            ),
        },
    )

    datamodule = TrainingDataModule(root="../dataset/temp")
    )
    train_supervised(
        task,
        datamodule,
        log_dirpath="../logs",
        study_name="training",
        experiment_name="base_potsdam-rgb_batch-size-2_max-epochs-100_new-loss_all-load_batch-size-16",
        max_epochs=5,
    )
