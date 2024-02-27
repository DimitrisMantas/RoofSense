from typing import Optional

import torchgeo.trainers
from torch.optim import AdamW
from torch.optim.lr_scheduler import (SequentialLR,
                                      CosineAnnealingWarmRestarts,
                                      ConstantLR, )
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassF1Score,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision,
                                         MulticlassRecall, )


class TrainingTask(torchgeo.trainers.SemanticSegmentationTask):
    def __init__(self, *args, tmax: int = 50, eta_min: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        # noinspection PyArgumentEqualDefault
        metrics = MetricCollection(
            {
                "AverageAccuracy": MulticlassAccuracy(
                    num_classes=num_classes, average="macro", ignore_index=ignore_index
                ),
                "OverallAccuracy": MulticlassAccuracy(
                    num_classes=num_classes, average="micro", ignore_index=ignore_index
                ),
                "AveragePrecision": MulticlassPrecision(
                    num_classes=num_classes, average="macro", ignore_index=ignore_index
                ),
                "AverageRecall": MulticlassRecall(
                    num_classes=num_classes, average="macro", ignore_index=ignore_index
                ),
                "AverageF1Score": MulticlassF1Score(
                    num_classes=num_classes, average="macro", ignore_index=ignore_index
                ),
                "AverageIoU": MulticlassJaccardIndex(
                    num_classes=num_classes, average="macro", ignore_index=ignore_index
                ),
                "WeightedIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    average="weighted",
                    ignore_index=ignore_index,
                ),
                # "AverageSpecificity": MulticlassSpecificity(
                #     num_classes=num_classes, average="macro", ignore_index=ignore_index
                # ),
                # "Kappa": MulticlassCohenKappa(
                #     num_classes=num_classes, ignore_index=ignore_index
                # ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        tmax: int = self.hparams["tmax"]
        eta_min: float = self.hparams["eta_min"]

        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])

        max_epochs = 1000
        warmup_epochs = int(max_epochs * 0.1)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                ConstantLR(optimizer,
                           # Keep the LR actually constant.
                           factor=1,
                           total_iters=warmup_epochs),
                # TODO: Increase T0 by a fixed amount at each restart
                # https://discuss.pytorch.org/t/using-cosineannealinglr-to-adjust-lr-within-single-epoch/42875
                # this can be done by incrementing a variable or something each time
                #  epoch % self.T_0 == 0
                # example:
                # self.T_0_=T_0 # init
                # if   epoch % self.T_0 == 0: self.T_0_*=0.1
                # self.T_cur = self.T_0_
                # and we can also mess with the restart LR in a similar way
                CosineAnnealingWarmRestarts(optimizer, T_0=tmax, eta_min=eta_min),
            ],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
