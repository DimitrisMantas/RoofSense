from dataclasses import dataclass
from typing import Any, Literal

import segmentation_models_pytorch as smp
import timm.layers
import torch


@dataclass(frozen=True, slots=True)
class TrainingTaskHyperparameterTuningConfig:
    # Augmentations
    append_lab: bool = False
    append_tgi: bool = False
    # Encoder
    encoder: Literal["tu-resnet18", "tu-resnet18d"] = "tu-resnet18"
    global_pool: str = "avg"
    aa_layer: bool = False
    drop_rate: float = 0
    drop_path_rate: float = 0
    zero_init_last: bool = False
    attn_layer: str | None = None
    # Decoder
    decoder_atrous_rate1: int = 6
    decoder_atrous_rate2: int = 12
    decoder_atrous_rate3: int = 18
    # Loss
    label_smoothing: float = 0
    # Optimizer
    optimizer: Literal["Adam", "AdamW"] = "Adam"
    lr: float = 1e-3
    beta2: float = 0.999
    weight_decay: float = 0
    # LR Scheduler
    lr_scheduler: Literal["CosineAnnealingLR", "PolynomialLR"] = "PolynomialLR"
    warmup_epochs: int = 0

    # This should be a cached property, but using slots means that there is no underlying dictionary to store the returned value.
    # The configuration is only meant to be used once anyway, so this is probably fine.
    @property
    def eps(self) -> float:
        return 1e-7 if self.optimizer == "Adam" else 1e-8

    @property
    def power(self) -> float | None:
        return 0.9 if self.lr_scheduler == "PolynomialLR" else None


def create_model(config: TrainingTaskHyperparameterTuningConfig) -> torch.nn.Module:
    return smp.create_model(
        arch="deeplabv3plus",
        encoder_name=config.encoder,
        in_channels=7 + 3 * config.append_lab + config.append_tgi,
        classes=8 + 1,
        global_pool=config.global_pool,
        aa_layer=timm.layers.BlurPool2d if config.aa_layer else None,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        zero_init_last=config.zero_init_last,
        block_args=dict(attn_layer=config.attn_layer),
        decoder_atrous_rates=(
            config.decoder_atrous_rate1,
            config.decoder_atrous_rate2,
            config.decoder_atrous_rate3,
        ),
        decoder_aspp_dropout=0,
    )


# https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
def configure_weight_decay_parameter_groups(
    model: torch.nn.Module,
) -> list[dict[str, Any]]:
    # Separate parameters.
    enabled_weight_decay = set()
    disabled_weight_decay = set()

    whitelist_modules = (torch.nn.Conv1d, torch.nn.Conv2d)
    blacklist_modules = (
        torch.nn.BatchNorm2d,
        # This is required for global context attention.
        timm.layers.norm.LayerNorm2d,
    )
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters():
            full_param_name = (
                f"{module_name}.{param_name}" if module_name else param_name
            )
            if param_name.endswith("bias"):
                disabled_weight_decay.add(full_param_name)
            elif param_name.endswith("weight") and isinstance(
                module, whitelist_modules
            ):
                enabled_weight_decay.add(full_param_name)
            elif param_name.endswith("weight") and isinstance(
                module, blacklist_modules
            ):
                # Normalization Scale
                disabled_weight_decay.add(full_param_name)

    # Validate split.
    params = {name: param for name, param in model.named_parameters()}
    params_in_both_groups = enabled_weight_decay & disabled_weight_decay
    params_in_none_groups = enabled_weight_decay | disabled_weight_decay
    assert len(params_in_both_groups) == 0
    assert len(params.keys() - params_in_none_groups) == 0

    return [
        {
            "params": [
                params[param_name] for param_name in sorted(list(enabled_weight_decay))
            ]
        },
        {
            "params": [
                params[param_name] for param_name in sorted(list(disabled_weight_decay))
            ],
            "weight_decay": 0,
        },
    ]
