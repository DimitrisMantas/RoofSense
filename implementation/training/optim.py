from typing import cast

import numpy as np
import optuna
import torch
from lightning import Callback
from optuna_integration import PyTorchLightningPruningCallback

from implementation.training.utils import (
    TrainingTaskConfig,
    configure_weight_decay_parameter_groups,
    create_model,
)
from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask


def objective(trial: optuna.Trial) -> float:
    config = TrainingTaskConfig(
        # Encoder
        # https://arxiv.org/pdf/1812.01187
        encoder="tu-resnet18d",
        # https://arxiv.org/pdf/2110.00476
        drop_path_rate=trial.suggest_float(name="drop_path_rate", low=0.05, high=0.1),
        # https://resolver.tudelft.nl/uuid:c463e920-61e6-40c5-89e9-25354fadf549
        attn_layer="eca",
        # Decoder
        decoder_atrous_rates=_suggest_decoder_atrous_rates(trial),
        # Loss
        # https://arxiv.org/pdf/1812.01187
        # https://arxiv.org/pdf/2110.00476
        label_smoothing=0.1,  # Optimizer
        # https://arxiv.org/pdf/1812.01187
        # https://arxiv.org/pdf/2110.00476
        optimizer="AdamW",
        # Search space adapted from https://arxiv.org/pdf/2110.00476.
        # The upper bound has been rounded up to the nearest power of 10.
        lr=trial.suggest_float(name="lr", low=0.001, high=0.01),
        # Search space adapted from https://arxiv.org/pdf/2110.00476.
        weight_decay=trial.suggest_float(name="weight_decay", low=0.01, high=0.05),
        # LR Scheduler
        # https://arxiv.org/pdf/1812.01187
        # https://arxiv.org/pdf/2110.00476
        scheduler="CosineAnnealingLR",
        # Search space adapted from https://arxiv.org/pdf/2110.00476.
        # The upper bound has been set to 50% of the total training time to account for potential instabilities to the small batch size used for training.
        warmup_epochs=trial.suggest_int(name="warmup_epochs", low=5, high=150),
    )

    cache = _check_trial_completed(trial)
    if cache is not None:
        return cache

    model = create_model(config)

    task = TrainingTask(
        model=model,
        loss_cfg={
            "names": ["crossentropyloss", "diceloss"],
            "weight": torch.from_numpy(
                np.fromfile(r"C:\Documents\RoofSense\roofsense\dataset\weights.bin")
            ).to(torch.float32),
            "include_background": False,
        },
        optimizer_cfg={
            "params": configure_weight_decay_parameter_groups(model),
            "eps": config.eps,
        },
        scheduler_cfg={"total_iters": 300},
    )

    datamodule = TrainingDataModule(root=r"C:\Documents\RoofSense\roofsense\dataset")

    trainer = train_supervised(
        task,
        datamodule,
        log_dirpath=r"C:\Documents\RoofSense\logs\3dgeoinfo",
        study_name=trial.study.study_name,
        experiment_name=trial.number,
        callbacks=cast(
            Callback, PyTorchLightningPruningCallback(trial, monitor=task.monitor_optim)
        ),
        max_epochs=300,
        test=False,
    )

    return trainer.callback_metrics[task.monitor_optim].item()


def _check_trial_completed(trial: optuna.Trial) -> float | None:
    # Check whether a given trial has already been completed and return the corresponding cached objective value from the underlying study storage instead of rerunning it.
    # This is useful when resuming deterministic studies.
    completed_trials = trial.study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    for t in reversed(completed_trials):
        if trial.params == t.params:
            return t.value


def _suggest_decoder_atrous_rates(trial: optuna.Trial) -> tuple[int, int, int]:
    # Search space adapted from https://arxiv.org/abs/1802.02611.
    # The upper bound has been selected as the highest base atrous rate encountered in relevant literature.
    decoder_atrous_rate1 = trial.suggest_int(
        name="decoder_atrous_rate1", low=6, high=12
    )
    # Various relationships between the atrous rates are explored: from r3=1+r2=1+r1 (i.e., linear progression) to r3=2*r2=2*r1 (i.e., geometric progression).
    decoder_atrous_rate2 = trial.suggest_int(
        name="decoder_atrous_rate2",
        low=decoder_atrous_rate1 + 1,
        high=decoder_atrous_rate1 * 2,
    )
    decoder_atrous_rate3 = trial.suggest_int(
        name="decoder_atrous_rate3",
        low=decoder_atrous_rate2 + 1,
        high=decoder_atrous_rate2 * 2,
    )
    return decoder_atrous_rate1, decoder_atrous_rate2, decoder_atrous_rate3


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///C:/Documents/RoofSense/logs/3dgeoinfo/hptuning/storage.db",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
        study_name="hptuning",
        direction=TrainingTask.monitor_optim_direction,
    )
    study.optimize(objective, timeout=24 * 60 * 60)
