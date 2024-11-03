from typing import cast

import numpy as np
import optuna
import timm.layers
import torch
from lightning import Callback

from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask


def objective(trial: optuna.Trial):
    # Input Data
    # This is fixed from the first round.
    append_hsv = False
    append_tgi = False

    # Decoder
    # This is fixed from the first round.
    base_atrous_rate = trial.suggest_int(name="base_atrous_rate", low=6, high=20)

    # Encoder
    # NOTE: This is dropped from the manual search.
    aa_layer = False
    # NOTE: This is fixed from the first round.
    attn_layer = "eca"
    # NOTE: This is fixed from the manual search.
    encoder = "resnet18d"

    # Regularisation
    # 10 Values
    drop_path_rate = trial.suggest_float(name="drop_path_rate", low=0, high=1e-1)
    # NODE: This is fixed from the manual search.
    label_smoothing = 0.1

    weight_decay = trial.suggest_float(name="weight_decay", low=0, high=1e-2)

    # Optimization Algorithm
    # NOTE: This is fixed from the second round.
    optimizer = "adamw"

    # Learning Rate Scheduling
    warmup_epochs = trial.suggest_int(name="warmup_epochs", low=50, high=150)
    # NOTE: This is fixed from the second round.
    annealing = "cos"

    # Nuisance Parameters
    # This is adjusted from the first round.
    lr = trial.suggest_float(name="lr", low=5e-4, high=5e-3)

    # Do not rerun the same experiment twice.
    completed_trials = trial.study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    for t in reversed(completed_trials):
        if trial.params == t.params:
            return t.value

    task = TrainingTask(
        in_channels=7 + 3 * append_hsv + append_tgi,
        decoder="deeplabv3plus",
        encoder=encoder,
        model_params={
            "decoder_atrous_rates": (
                base_atrous_rate,
                base_atrous_rate * 2,
                base_atrous_rate * 3,
            ),
            "encoder_params": {
                "block_args": {"attn_layer": attn_layer},
                "aa_layer": timm.layers.BlurPool2d if aa_layer else None,
                "drop_path_rate": drop_path_rate,
            },
        },
        loss_params={
            "names": ["crossentropyloss", "diceloss"],
            "weight": torch.from_numpy(
                np.fromfile(r"C:\Documents\RoofSense\dataset\temp\weights_tf-idf.bin")
            ).to(torch.float32),
            "include_background": False,
            "label_smoothing": label_smoothing,
        },
        optimizer=optimizer,
        lr=lr,
        warmup_epochs=warmup_epochs,
        annealing=annealing,
        weight_decay=weight_decay,
    )

    datamodule = TrainingDataModule(
        root="../../dataset/temp", append_hsv=append_hsv, append_tgi=append_tgi
    )

    trainer = train_supervised(
        task,
        datamodule,
        log_dirpath="../../logs",
        study_name="optimization_random_search_round_3_tpe",
        experiment_name=trial.number,
        callbacks=cast(
            Callback,
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val/MacroIoU"
            ),
        ),
        test=False,
        max_epochs=200,
    )

    return trainer.callback_metrics["val/MacroIoU"].item()


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optimization_random_search_round_3_tpe.db",
        # sampler=optuna.samplers.RandomSampler(seed=0),
        sampler=optuna.samplers.TPESampler(seed=0),
        # pruner=optuna.pruners.HyperbandPruner(min_resource=20, reduction_factor=2),
        # pruner=optuna.pruners.MedianPruner(
        #     n_startup_trials=10, n_warmup_steps=30, interval_steps=5
        # ),
        pruner=optuna.pruners.NopPruner(),
        study_name="optimization_random_search_round_3",
        direction="maximize",
    )
    study.optimize(objective, n_trials=50)
