from dataclasses import dataclass
from typing import Literal

import timm.layers


@dataclass(frozen=True, slots=True)
class TrainingTaskConfig:
    """Grouped configuration options for the training task to simplify the hyperparameter optimization process."""

    # Augmentations
    append_hsv: bool = False
    append_tgi: bool = False
    # Encoder
    encoder: Literal["tu-resnet18", "tu-resnet18d"] = "tu-resnet18"
    aa_layer: type[timm.layers.BlurPool2d] | None = None
    drop_path_rate: float = 0
    attn_layer: Literal["eca", "se"] | None = None
    # Decoder
    decoder_min_atrous_rate: int = 6
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
        object.__setattr__(self, "eps", 1e-7 if self.optimizer == "Adam" else 1e-8)
        # if self.scheduler == "PolynomialLR":
        #     object.__setattr__(self,"power",0.9)
