from dataclasses import dataclass
from typing import Literal

import timm.layers


@dataclass(frozen=True, slots=True)
class Config:
    aa_layer: type[timm.layers.BlurPool2d] | None = None
    annealing: Literal["cos", "poly"] = "poly"
    append_hsv: bool = False
    append_tgi: bool = False
    attn_layer: Literal["eca", "se"] | None = None
    base_atrous_rate: int = 6
    drop_path_rate: float = 0
    encoder: Literal["resnet18", "resnet18d"] = "resnet18"
    label_smoothing: float = 0
    learning_rate: float = 1e-3
    optimizer: Literal["adam", "adamw"] = "adam"
    warmup_epochs: int = 0
    weight_decay: float = 0
