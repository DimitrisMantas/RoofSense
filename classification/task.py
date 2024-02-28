from __future__ import annotations

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
from torchgeo.models import FCN
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
        T_0: int = 50,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        ignore_metrics: Optional[
            PerformanceMetric | dict[str, PerformanceMetric]
        ] = None,
        max_warmup_epochs: int = 40,
        warmup_epoch_pct: float = 0.05,
        init_eta_factor: float = 0.1,
        max_epochs: int = 1000,
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

        max_epochs: int = self.hparams["num_epochs"]
        if self.trainer and self.trainer.max_epochs:
            max_epochs = self.trainer.max_epochs
        warmup_epochs = min(
            int(max_epochs * self.hparams["warmup_epoch_pct"]),
            self.hparams["max_warmup_epochs"],
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=warmup_epochs * self.hparams["init_eta_factor"],
                    total_iters=warmup_epochs,
                ),
                # TODO: Check whether having a decaying restart learning rate is
                #  possible.
                CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.hparams["T_0"],
                    T_mult=self.hparams["T_mult"],
                    eta_min=self.hparams["eta_min"],
                ),
            ],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lrs = []

    model = FCN(in_channels=6, classes=10)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # TODO: Expose this parameter as an initializer argument.
    warmup_epochs = min(int(1000 * 0.05), 50)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            ),
            # TODO: Check whether having a decaying restart learning rate is
            #  possible.
            CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,
                T_mult=2,
                eta_min=1e-6,
            ),
        ],
        milestones=[warmup_epochs],
    )
    for epoch in range(1000):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    plt.plot(range(1000), lrs)
    plt.show()
