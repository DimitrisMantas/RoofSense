from __future__ import annotations

import os
import warnings
from typing import Any, Literal

import segmentation_models_pytorch as smp
import torchgeo.trainers.utils
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.models import FCN
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import ClasswiseWrapper, Metric, MetricCollection
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassF1Score,
                                         MulticlassJaccardIndex, )
from torchvision.models import WeightsEnum
from typing_extensions import override

from training.loss import CompoundLoss


class TrainingTask(SemanticSegmentationTask):
    def __init__(
        self,
        *args,
        # The total number of warmup epochs, expressed as a percentage of the maximum
        # number of training epochs, as specified by the trainer this task is associated
        # with or `max_epochs`.
        warmup_time: float = 0.05,
        # The maximum number of warmup epochs.
        max_warmup_epochs: int = 50,
        # The total number of epochs constituting the period of the first cycle of the
        # annealing phase.
        T_0: int = 50,
        # The period ratio `Ti+1/Ti` of two consecutive cycles `i` of the annealing
        # phase.
        T_mult: int = 2,
        # The maximum number of training epochs.
        # NOTE: This parameter is used to provide a fallback in case the trainer this
        # task is used with cannot provide it.
        max_epochs: int = 1000,
        # The learning rate at the end of the warmup and each new cycle of the
        # subsequent annealing phases. This parameter is henceforth referred to
        # as the "nominal learning rate".
        lr: float = 1e-4,
        # The learning rate at the start of the warmup phase, expressed as a
        # percentage of the nominal learning rate.
        init_lr_pct: float = 0.1,
        # The minimum learning rate at the annealing phase, expressed as a percentage
        # of the nominal learning rate.
        min_lr_pct: float = 0.01,
        model_kwargs: dict[str, float | str | None] | None = None,
        loss,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @override
    def configure_models(self) -> None:
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]
        num_filters: int = self.hparams["num_filters"]

        standard_kwargs = {
            "encoder_name": backbone,
            "encoder_weights": "imagenet" if weights is True else None,
            "in_channels": in_channels,
            "classes": num_classes,
        }
        optional_kwargs: dict[str, float | str | None] | None = self.hparams[
            "model_kwargs"
        ]

        # TODO: Initialize any SMP model dynamically.
        if model == "unet":
            if optional_kwargs is None:
                self.model = smp.Unet(**standard_kwargs)
            else:
                self.model = smp.Unet(**standard_kwargs, **optional_kwargs)
        elif model == "deeplabv3+":
            if optional_kwargs is None:
                self.model = smp.DeepLabV3Plus(**standard_kwargs)
            else:
                self.model = smp.DeepLabV3Plus(**standard_kwargs, **optional_kwargs)
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if model != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = torchgeo.trainers.utils.extract_backbone(weights)
                else:
                    state_dict = torchgeo.models.get_weight(weights).get_state_dict(
                        progress=True
                    )
                self.model.encoder.conv1 = (
                    torchgeo.trainers.utils.reinit_initial_conv_layer(
                        self.model.encoder.conv1,
                        new_in_channels=state_dict["conv1.weight"].shape[1],
                        keep_rgb_weights=True,
                    )
                )
                self.model.encoder.load_state_dict(state_dict)
                self.model.encoder.conv1 = (
                    torchgeo.trainers.utils.reinit_initial_conv_layer(
                        self.model.encoder.conv1,
                        new_in_channels=in_channels,
                        keep_rgb_weights=True,
                    )
                )

        # Freeze backbone
        if self.hparams["freeze_backbone"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    @override
    def configure_losses(self) -> None:
        self.criterion = CompoundLoss(**self.hparams.loss)

    @override
    def configure_metrics(self) -> None:
        self.train_metrics_scalar = MetricCollection(
            self._init_metrics(average="macro")
            | self._init_metrics(average="micro"),  # | classwise,
            prefix="tra/",
        )
        self.val_metrics_scalar = self.train_metrics_scalar.clone(prefix="val/")
        self.test_metrics_scalar = self.train_metrics_scalar.clone(prefix="tst/")

        temp = self._init_metrics(average=None)
        # TODO: Figure out why IoU cannot be wrapped.
        temp.pop("IoU")
        self.train_metrics_class = MetricCollection(
            {
                name: ClasswiseWrapper(metric, prefix=f"{name}/")
                for name, metric in temp.items()
            },
            prefix="tra/",
        )
        self.val_metrics_class = self.train_metrics_class.clone(prefix="val/")
        self.test_metrics_class = self.train_metrics_class.clone(prefix="tst/")

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        # TODO: Explore different optimizers and schedulers and their specific
        #  parameters.
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])

        max_epochs: int = self.hparams["max_epochs"]
        if self.trainer and self.trainer.max_epochs:
            warnings.warn(
                "The trainer does not specify a maximum number of epochs", UserWarning
            )
            max_epochs = self.trainer.max_epochs
        warmup_epochs = min(
            int(max_epochs * self.hparams["warmup_time"]),
            self.hparams["max_warmup_epochs"],
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=self.hparams["init_lr_pct"],
                    total_iters=warmup_epochs,
                ),
                # TODO: Check whether having a decaying restart learning rate is
                #  possible.
                CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.hparams["T_0"],
                    T_mult=self.hparams["T_mult"],
                    eta_min=self.hparams["lr"] * self.hparams["min_lr_pct"],
                ),
            ],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def _init_metrics(
        self, average: Literal["macro", "micro", "none"] | None
    ) -> dict[str, Metric]:
        _average = "multiclass" if average == "none" or average is None else average
        _average.capitalize()

        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = self.hparams["loss"]["ignore_index"]

        return {
            f"{_average}Accuracy": MulticlassAccuracy(
                num_classes,
                average=average,
                ignore_index=ignore_index,
                multidim_average="global",
            ),
            f"{_average}F1Score": MulticlassF1Score(
                num_classes,
                average=average,
                ignore_index=ignore_index,
                multidim_average="global",
            ),
            f"{_average}IoU": MulticlassJaccardIndex(
                num_classes, average=average, ignore_index=ignore_index
            ),
        }