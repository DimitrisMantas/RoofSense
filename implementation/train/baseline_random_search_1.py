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
    append_hsv = trial.suggest_categorical(name="append_hsv", choices=[True, False])
    append_tgi = trial.suggest_categorical(name="append_tgi", choices=[True, False])

    # Decoder
    # 5 Values
    base_atrous_rate = trial.suggest_int(
        name="base_atrous_rate",
        low=1,
        # This value has been adjusted from 20 in order to accommodate the necessary
        # step.
        high=21,
        step=5,
    )

    # Encoder
    # NOTE: This is dropped from the manual search.
    aa_layer = False
    attn_layer = trial.suggest_categorical(
        name="attn_layer",
        choices=[
            "eca",  # NOTE: This is dropped from the manual search.
            # "se",
            None,
        ],
    )
    # NOTE: This is fixed from the manual search.
    encoder = "resnet18d"

    # Regularisation
    # 10 Values
    drop_path_rate = trial.suggest_float(
        name="drop_path_rate", low=0, high=1e-1, step=1e-2
    )
    # NODE: This is fixed from the manual search.
    label_smoothing = trial.suggest_float(
        name="label_smoothing", low=0.05, high=0.15, step=1e-2
    )
    # 10 Values
    weight_decay = trial.suggest_float(name="weight_decay", low=0, high=1e-2, step=1e-3)

    # Nuisance Parameters
    # 10 Values
    lr = trial.suggest_float(name="lr", low=5e-4, high=5e-3, step=5e-4)

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
        lr=lr,
        weight_decay=weight_decay,
    )

    datamodule = TrainingDataModule(
        root="../../dataset/temp", append_hsv=append_hsv, append_tgi=append_tgi
    )

    trainer = train_supervised(
        task,
        datamodule,
        log_dirpath="../../logs",
        study_name=trial.study.study_name,
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
        storage="sqlite:///optimization_random_search_round_1.db",
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
        study_name="optimization_random_search_round_1",
        direction="maximize",
    )
    study.optimize(objective, n_trials=50)
