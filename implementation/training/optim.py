from typing import Any, cast

import numpy as np
import optuna
import segmentation_models_pytorch as smp
import torch
from lightning import Callback

from implementation.training.utils import TrainingTaskConfig
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
        attn_layer="eca",  # Decoder
        decoder_atrous_rates=suggest_decoder_atrous_rates(trial),
        # Loss
        # https://arxiv.org/pdf/1812.01187
        # https://arxiv.org/pdf/2110.00476
        label_smoothing=0.1,
        # Optimizer
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

    value = check_trial_completed(trial)
    if value is not None:
        return value

    model = create_suggested_model(config)

    task = TrainingTask(
        model=model,
        loss_cfg={
            "names": ["crossentropyloss", "diceloss"],
            "weight": torch.from_numpy(
                np.fromfile(r"/roofsense/dataset/weights.bin")
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
            Callback,
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor=task.monitor_optim
            ),
        ),
        max_epochs=300,
        test=False,
    )

    return trainer.callback_metrics[task.monitor_optim].item()


def suggest_decoder_atrous_rates(trial: optuna.Trial) -> tuple[int, int, int]:
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


def check_trial_completed(trial: optuna.Trial) -> float | None:
    # Do not rerun the same experiment twice.
    completed_trials = trial.study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    for t in reversed(completed_trials):
        if trial.params == t.params:
            return t.value


def configure_weight_decay_parameter_groups(
    model: torch.nn.Module,
) -> list[dict[str, Any]]:
    # Weight Decay
    # https://arxiv.org/pdf/1812.01187
    weights = []
    other = []
    for name, param in model.named_parameters():
        if "weight" in name and not isinstance(param, torch.nn.BatchNorm2d):
            weights.append(param)
        else:
            other.append(param)
    params = [{"params": weights}, {"params": other, "weight_decay": 0}]
    return params


def create_suggested_model(config: TrainingTaskConfig) -> torch.nn.Module:
    return smp.create_model(
        arch="deeplabv3plus",
        encoder_name=config.encoder,
        in_channels=7,
        classes=9,
        drop_path_rate=config.drop_path_rate,
        block_args=dict(attn_layer=config.attn_layer),
        decoder_atrous_rates=config.decoder_atrous_rates,
        decoder_aspp_dropout=0,
    )
