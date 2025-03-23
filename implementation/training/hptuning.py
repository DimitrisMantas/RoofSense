import logging
import os
from typing import cast

import numpy as np
import optuna
import torch
from lightning import Callback
from optuna_integration import PyTorchLightningPruningCallback

from implementation.training.baseline import main
from implementation.training.utils import (
    TrainingTaskConfig,
    configure_weight_decay_parameter_groups,
    create_model,
)
from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask


def objective(trial: optuna.Trial) -> float:
    # https://arxiv.org/pdf/1812.01187
    # https://arxiv.org/pdf/2110.00476
    # https://arxiv.org/pdf/2201.03545
    # https://arxiv.org/abs/2301.00808
    config = TrainingTaskConfig(
        # Encoder
        encoder="tu-resnet18d",
        drop_path_rate=trial.suggest_float(
            name="drop_path_rate",
            low=0,
            # This covers all sources except some tables in https://arxiv.org/pdf/2201.03545 and https://arxiv.org/abs/2301.00808 which set it to [0.2, 0.5].
            high=0.1,
        ),
        # This proved to be useful in https://resolver.tudelft.nl/uuid:c463e920-61e6-40c5-89e9-25354fadf549.
        attn_layer="eca",
        # Loss
        # This covers all sources except one table in https://arxiv.org/abs/2301.00808 which sets it to 0.2.
        label_smoothing=0.1,
        # Optimizer
        optimizer="AdamW",
        lr=trial.suggest_float(name="lr", low=1e-4, high=0.01, log=True),
        weight_decay=trial.suggest_float(name="weight_decay", low=0, high=0.05),
        # LR Scheduler
        lr_scheduler="CosineAnnealingLR",
        warmup_epochs=trial.suggest_int(
            name="warmup_epochs",
            low=0,
            # All sources set this to max(5% of the total training duration, 5) except one table in https://arxiv.org/abs/2301.00808 which sets it 20/300 epochs (~6.67% of the total training duration).
            #
            # The maximum warmup duration should be at most equal to 10% of the total training duration.
            # However, if more than 5% of training is eventually performed under a warmup regime, then the total training durations might need to be further increased.
            # https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq#how_to_apply_learning_rate_warmup
            high=40,
        ),
    )

    value = _lookup_objective_value(trial)
    if value is not None:
        return value

    # FIXME: hparams.yaml is wrong when passing custom models.
    # FIXME: hparams.yaml includes full parameter groups (> 20 MB) when they are specified.
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
            "params": configure_weight_decay_parameter_groups(model),
            "lr": config.lr,
            "eps": config.eps,
            "weight_decay": config.weight_decay,
        },
        lr_scheduler=config.lr_scheduler,
        lr_scheduler_cfg={"T_max": 400},
        warmup_epochs=config.warmup_epochs,
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
        # The warmup duration is additional to the annealing duration.
        # https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq#how_to_apply_learning_rate_warmup
        max_epochs=400 + config.warmup_epochs,
        test=False,
    )

    return trainer.callback_metrics[task.monitor_optim].item()


def _lookup_objective_value(trial: optuna.Trial) -> float | None:
    # Check whether a given trial has already been completed and return the corresponding cached objective value from the underlying study storage instead of rerunning it.
    # This is useful when resuming deterministic studies.
    completed_trials = trial.study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    for t in reversed(completed_trials):
        if trial.params == t.params:
            return t.value


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Establish the baseline.
    main()

    # Perform hyperparameter tuning.
    study_name = "hptuning"
    log_dirpath = os.path.join(r"C:\Documents\RoofSense\logs\3dgeoinfo", study_name)

    os.makedirs(log_dirpath, exist_ok=True)

    storage = f"sqlite:///{log_dirpath}/storage.db"
    sampler = optuna.samplers.GPSampler(seed=0)
    # TODO: Consider using a pruner.
    pruner = optuna.pruners.NopPruner()
    direction = TrainingTask.monitor_optim_direction

    try:
        study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            direction=direction,
        )
    except optuna.exceptions.DuplicatedStudyError:
        # Load the study with any stale trials removed.
        # TODO: This feature is experimental!
        study = optuna.load_study(study_name=study_name, storage=storage)
        trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        optuna.delete_study(study_name=study_name, storage=storage)
        study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            direction=direction,
        )
        study.add_trials(trials)

    study.optimize(objective, n_trials=max(0, 50 - len(study.trials)))
