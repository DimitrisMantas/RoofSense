from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import segmentation_models_pytorch as smp
import torch


# TODO: Make this class frozen and add slots.
@dataclass
class TrainingTaskConfig:
    """Grouped configuration options for the training task to simplify the hyperparameter optimization process."""

    # Encoder
    encoder: Literal["tu-resnet18", "tu-resnet18d"] = "tu-resnet18"
    drop_path_rate: float = 0
    attn_layer: Literal["eca", "se"] | None = None
    # Decoder
    decoder_atrous_rates: Iterable[int] = (6, 12, 18)
    # Loss
    label_smoothing: float = 0
    # Optimizer
    optimizer: Literal["Adam", "AdamW"] = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0
    # LR Scheduler
    scheduler: Literal["CosineAnnealingLR", "PolynomialLR"] = "PolynomialLR"
    warmup_epochs: int = 0

    def __post_init__(self) -> None:
        self.eps = 1e-7 if self.optimizer == "Adam" else 1e-8


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


def create_model(config: TrainingTaskConfig) -> torch.nn.Module:
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
