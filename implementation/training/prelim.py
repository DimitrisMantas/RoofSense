import logging

import numpy as np
import torch

from implementation.training.utils import (
    TrainingTaskHyperparameterTuningConfig,
    create_model,
)
from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask


def main():
    # config = TrainingTaskHyperparameterTuningConfig(append_hsv=True)

    for experiment_name in ["avgmax", "catavgmax", "max"]:
        config = TrainingTaskHyperparameterTuningConfig(global_pool=experiment_name)

        model = create_model(config)

        task = TrainingTask(
            model=model,
            loss_cfg={
                "names": ["crossentropyloss", "diceloss"],
                "weight": torch.from_numpy(
                    np.fromfile(r"C:\Documents\RoofSense\roofsense\dataset\weights.bin")
                ).to(torch.float32),
                "include_background": False,
                "label_smoothing": config.label_smoothing,
            },
            optimizer=config.optimizer,
            optimizer_cfg={
                "lr": config.lr,
                "betas": (0.9, config.beta2),
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
            lr_scheduler=config.lr_scheduler,
            lr_scheduler_cfg={"total_iters": 400, "power": config.power},
            warmup_epochs=config.warmup_epochs,
        )

        datamodule = TrainingDataModule(
            root=r"C:\Documents\RoofSense\roofsense\dataset",
            append_lab=config.append_lab,
            append_tgi=config.append_tgi,
        )

        train_supervised(
            task,
            datamodule,
            log_dirpath=r"C:\Documents\RoofSense\logs\3dgeoinfo\prelim",
            study_name="global_pool",
            experiment_name=experiment_name,
            # The warmup duration is additional to the annealing duration.
            # https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq#how_to_apply_learning_rate_warmup
            max_epochs=400 + config.warmup_epochs,
            test=False,
        )


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    main()
