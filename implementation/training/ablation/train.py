import logging
import os

import numpy as np
import optuna
import torch

from implementation.training.utilities import (
    TrainingTaskHyperparameterTuningConfig,
    configure_weight_decay_parameter_groups,
    create_model,
)
from roofsense.enums.band import Band
from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask


def main():
    study_name = "optimization"
    optim_log_dirpath = os.path.join(r"/logs/3dgeoinfo", study_name)

    study = optuna.load_study(
        study_name="optim", storage=f"sqlite:///{optim_log_dirpath}/storage.db"
    )

    best_params = study.best_params
    # Convert parameter format.
    for param in ["lab", "tgi"]:
        best_params[f"append_{param}"] = best_params.pop(param)

    for ablation, bands in zip(
        ["density", "lidar", "ndrm", "reflectance", "slope"],
        [
            [0, 1, 2, 3, 4, 5],  # density
            [0, 1, 2],  # lidar
            [0, 1, 2, 3, 4, 6],  # ndrm
            [0, 1, 2, 4, 5, 6],  # reflectance
            [0, 1, 2, 3, 5, 6],  # slope
        ],
    ):
        config = TrainingTaskHyperparameterTuningConfig(
            # Add constant parameters.
            # Encoder
            encoder="tu-resnet18d",
            zero_init_last=True,
            # Loss
            label_smoothing=0.1,
            # Optimizer
            optimizer="AdamW",
            # LR Scheduler
            lr_scheduler="CosineAnnealingLR",
            # Miscellaneous
            in_channels=len(bands),
            **best_params,
        )

        num_trials = 1 if ablation == "ndrm" else 3
        for _ in range(num_trials):
            model = create_model(config)

            task = TrainingTask(
                model=model,
                loss_cfg={
                    "names": ["crossentropyloss", "diceloss"],
                    "weight": torch.from_numpy(
                        np.fromfile(r"/roofsense/dataset/weights.bin")
                    ).to(torch.float32),
                    "include_background": False,
                    "label_smoothing": config.label_smoothing,
                },
                optimizer=config.optimizer,
                optimizer_cfg={
                    "params": configure_weight_decay_parameter_groups(model),
                    "lr": config.lr,
                    "betas": (0.9, config.beta2),
                    "eps": config.eps,
                    "weight_decay": config.weight_decay,
                },
                lr_scheduler=config.lr_scheduler,
                lr_scheduler_cfg={"T_max": 400},
                warmup_epochs=config.warmup_epochs,
            )

            datamodule = TrainingDataModule(
                root=r"C:\Documents\RoofSense\roofsense\dataset",
                bands=[
                    # Align band numbers with those that rasterio expects.
                    band + 1
                    for band in bands
                ],
                append_lab=config.append_lab,
                append_tgi=config.append_tgi,
                slice=bands,
            )

            train_supervised(
                task,
                datamodule,
                log_dirpath=os.path.join(
                    os.path.dirname(optim_log_dirpath), "ablation"
                ),
                study_name=ablation,
                # The warmup duration is additional to the annealing duration.
                # https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq#how_to_apply_learning_rate_warmup
                max_epochs=400 + config.warmup_epochs,
                test=False,
            )


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    main()
