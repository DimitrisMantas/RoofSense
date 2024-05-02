from __future__ import annotations

import warnings
from typing import Literal

from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassJaccardIndex,
                                         MulticlassF1Score, )


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def configure_metrics(self) -> None:
        scalar_metrics = MetricCollection(
            self._init_metrics(average="macro") | self._init_metrics(average="micro")
        )
        self.train_metrics = scalar_metrics.clone(prefix="train_")
        self.val_metrics = scalar_metrics.clone(prefix="val_")
        self.test_metrics = scalar_metrics.clone(prefix="test_")

    def configure_optimizers(
        self,
    ) -> OptimizerLRSchedulerConfig:
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

        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = self.hparams["ignore_index"]

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
