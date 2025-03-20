from collections.abc import Iterable
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch

from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask


def main(
    experiment_names: int | str | Iterable[str] | Iterable[str] = None,
    zero_init_bn_scale: bool = False,
    only_weight_weight_decay: bool = False,
):
    model = smp.create_model(
        arch="deeplabv3plus",
        encoder_name="tu-resnet18",
        encoder_weights="imagenet",
        in_channels=7,
        classes=9,
        decoder_atrous_rates=(6, 12, 18),
        decoder_aspp_dropout=0,
    )
    experiment_names = (
        [experiment_names]
        if isinstance(experiment_names, Iterable)
        else experiment_names
    )

    # https://arxiv.org/pdf/1812.01187
    # TODO: Add type hints.
    params: list[dict[str, Any]] | None = None
    if only_weight_weight_decay:
        weights = []
        other = []
        for name, param in model.named_parameters():
            if "weight" in name and not isinstance(param, torch.nn.BatchNorm2d):
                weights.append(param)
            else:
                other.append(param)
        params = [{"params": weights}, {"params": other, "weight_decay": 0}]

    task = TrainingTask(
        model=model,
        loss_cfg={
            "names": ["crossentropyloss", "diceloss"],
            "weight": torch.from_numpy(
                np.fromfile(r"C:\Documents\RoofSense\roofsense\dataset\weights.bin")
            ).to(torch.float32),
            "include_background": False,
        },
        optimizer_cfg={"params": params, "eps": 1e-7},
        scheduler_cfg={"total_iters": 300, "power": 0.9},
    )

    datamodule = TrainingDataModule(root=r"C:\Documents\RoofSense\roofsense\dataset")

    for name in experiment_names:
        train_supervised(
            task,
            datamodule,
            log_dirpath=r"C:\Documents\RoofSense\logs\3dgeoinfo",
            study_name="baseline",
            experiment_name=name,
            max_epochs=300,
            test=False,
        )


if __name__ == "__main__":
    # main(experiment_names="zero_init_bn_scale", zero_init_bn_scale=True)
    # This configuration should produce the same results as the baseline when using Adam with its default parameters.
    main(experiment_names="only_weight_weight_decay", only_weight_weight_decay=True)
