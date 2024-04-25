from __future__ import annotations

import warnings
from enum import StrEnum, auto, Flag, verify, UNIQUE
from functools import reduce
from operator import or_
from typing import Optional

import torchgeo.trainers
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import classproperty
from torch.optim import AdamW
from torch.optim.lr_scheduler import (SequentialLR,
                                      CosineAnnealingWarmRestarts,
                                      LinearLR, )
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassF1Score,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision,
                                         MulticlassRecall, )


@verify(UNIQUE)
class PerformanceMetric(Flag):
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    F1_SCORE = auto()
    IOU = auto()

    # noinspection PyPep8Naming,PyMethodParameters
    @classproperty
    def ALL(cls):
        return reduce(or_, cls)


@verify(UNIQUE)
class PerformanceMetricAverage(StrEnum):
    MICRO = auto()
    MACRO = auto()
    WEIGHTED = auto()


class TrainingTask(torchgeo.trainers.SemanticSegmentationTask):
    # noinspection PyUnusedLocal,PyPep8Naming
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
        ignore_metrics: Optional[
            PerformanceMetric | dict[str, PerformanceMetric]
        ] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _format_metric_name(
        self,
    ):
        # TODO: Move all metric name formatting in this function.
        pass

    def _init_metrics(
        self,
        average: PerformanceMetricAverage,
        ignore_metrics: Optional[PerformanceMetric] = None,
    ) -> dict[str, Metric]:
        # NOTE: This field name is provided by PyTorch Lightning and is fixed.
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]

        # noinspection PyTypeChecker
        metrics = {
            f"{average}Accuracy": MulticlassAccuracy(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            f"{average}Precision": MulticlassPrecision(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            f"{average}Recall": MulticlassRecall(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            f"{average}F1Score": MulticlassF1Score(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
            f"{average}IoU": MulticlassJaccardIndex(
                num_classes=num_classes,
                average=average,
                ignore_index=ignore_index,
            ),
        }

        if ignore_metrics is not None:
            for metric in list(ignore_metrics):
                if metric == PerformanceMetric.F1_SCORE:
                    metric_name_suffix = (
                        " ".join(metric.name.split("_")).title().replace(" ", "")
                    )
                elif metric == PerformanceMetric.IOU:
                    metric_name_suffix = metric.name.capitalize().replace("u", "U")
                else:
                    metric_name_suffix = metric.name.capitalize()
                # TODO: Check whether  a fallback value of None should be added to
                #  allow for silent key deletions.
                metrics.pop(f"{average}{metric_name_suffix}")

        return metrics

    # noinspection PyAttributeOutsideInit
    def configure_metrics(self) -> None:
        ignore_metrics = self.hparams["ignore_metrics"]
        if isinstance(ignore_metrics, PerformanceMetric):
            ignore_metrics = {
                average: ignore_metrics for average in PerformanceMetricAverage
            }

        metrics = {}
        for average in PerformanceMetricAverage:
            # noinspection PyTypeChecker
            metrics = {
                **metrics,
                **self._init_metrics(
                    average,
                    ignore_metrics=ignore_metrics.get(average, None)
                    if ignore_metrics is not None
                    else None,
                ),
            }

        # NOTE: These field names are provided by PyTorch Lightning and are fixed.
        # TODO: Finalize the metric name suffix for each training stage,
        self.train_metrics = MetricCollection(metrics, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

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
