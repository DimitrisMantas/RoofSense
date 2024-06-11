import math
import os
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torchgeo.trainers.utils
import torchseg
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.models import FCN
from torchgeo.trainers import SemanticSegmentationTask
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassConfusionMatrix,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision,
                                         MulticlassRecall,
                                         MulticlassSpecificity, )
from torchvision.models import WeightsEnum
from typing_extensions import override

from training.loss import CompoundLoss
from training.wrappers import MacroAverageWrapper
from utils.color import get_fg_color


class TrainingTask(SemanticSegmentationTask):
    # TODO: See if renaming the loss messes up something.
    monitor = "val/loss"

    def __init__(
        self,
        *args,
        # The total number of warmup epochs, expressed as a percentage of the maximum
        # number of training epochs, as specified by the trainer this task is associated
        # with or `max_epochs`.
        warmup_time: float = 0.05,
        # The maximum number of warmup epochs.
        max_warmup_epochs: int = 30,
        # The total number of epochs constituting the period of the first cycle of the
        # annealing phase.
        T_0: int = 60,
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
        init_lr_pct: float = 1 / 3,
        # The minimum learning rate at the annealing phase, expressed as a percentage
        # of the nominal learning rate.
        min_lr_pct: float = 0,
        model_kwargs: dict[str, dict[str,float|str|Sequence[float]|None]|float | str | None] | None = None,
        loss_params: dict[str, Any],
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
                self.model = torchseg.Unet(**standard_kwargs)
            else:
                self.model = torchseg.Unet(**standard_kwargs, **optional_kwargs)
        elif model == "deeplabv3+":
            if optional_kwargs is None:
                self.model = torchseg.DeepLabV3Plus(**standard_kwargs)
            else:
                self.model = torchseg.DeepLabV3Plus(**standard_kwargs, **optional_kwargs)
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
        self.criterion = CompoundLoss(**self.hparams.loss_params)

    @override
    def configure_metrics(self) -> None:
        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = (
            0 if self.hparams["loss_params"]["ignore_background"] else None
        )

        self.tra_metrics = MetricCollection(
            {  # Macro
                # "MacroAccuracy": MulticlassAccuracy(
                #     num_classes=num_classes,
                #     average="macro",
                #     ignore_index=ignore_index,
                #     # NOTE: Assign NaN to absent and ignored classes so that they can be
                #     # identified and excluded from the macroscopic averaging step.
                #     zero_division=torch.nan,
                # ),
                "MacroPrecision": MacroAverageWrapper(
                    MulticlassPrecision(
                        num_classes=num_classes,
                        average="none",
                        ignore_index=ignore_index,
                        # NOTE: Assign NaN to absent and ignored classes so that they can be
                        # identified and excluded from the macroscopic averaging step.
                        zero_division=torch.nan,
                    )
                ),
                "MacroRecall": MacroAverageWrapper(
                    MulticlassRecall(
                        num_classes=num_classes,
                        average="none",
                        ignore_index=ignore_index,
                        # NOTE: Assign NaN to absent and ignored classes so that they can be
                        # identified and excluded from the macroscopic averaging step.
                        zero_division=torch.nan,
                    )
                ),
                # "MacroSpecificity": MulticlassSpecificity(
                #     num_classes=num_classes,
                #     average="macro",
                #     ignore_index=ignore_index,
                #     # NOTE: Assign NaN to absent and ignored classes so that they can be
                #     # identified and excluded from the macroscopic averaging step.
                #     zero_division=torch.nan,
                # ),
                "MacroIoU": MacroAverageWrapper(
                    MulticlassJaccardIndex(
                        num_classes=num_classes,
                        average="none",
                        ignore_index=ignore_index,
                        # NOTE: Assign NaN to absent and ignored classes so that they can be
                        # identified and excluded from the macroscopic averaging step.
                        zero_division=torch.nan,
                    )
                ),  # Micro
                "MicroAccuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    average="micro",
                    ignore_index=ignore_index,
                    # NOTE: Assign NaN to absent and ignored classes so that they can be
                    # identified and excluded from the macroscopic averaging step.
                    zero_division=torch.nan,
                ),
                # "MicroPrecision": MulticlassPrecision(
                #     num_classes=num_classes,
                #     average="micro",
                #     ignore_index=ignore_index,
                #     # NOTE: Assign NaN to absent and ignored classes so that they can be
                #     # identified and excluded from the macroscopic averaging step.
                #     zero_division=torch.nan,
                # ),
                # "MicroRecall": MulticlassRecall(
                #     num_classes=num_classes,
                #     average="micro",
                #     ignore_index=ignore_index,
                #     # NOTE: Assign NaN to absent and ignored classes so that they can be
                #     # identified and excluded from the macroscopic averaging step.
                #     zero_division=torch.nan,
                # ),
                "MicroSpecificity": MulticlassSpecificity(
                    num_classes=num_classes,
                    average="micro",
                    ignore_index=ignore_index,
                    # NOTE: Assign NaN to absent and ignored classes so that they can be
                    # identified and excluded from the macroscopic averaging step.
                    zero_division=torch.nan,
                ),
                "MicroIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    average="micro",
                    ignore_index=ignore_index,
                    # NOTE: Assign NaN to absent and ignored classes so that they can be
                    # identified and excluded from the macroscopic averaging step.
                    zero_division=torch.nan,
                ),
            },
            prefix="tra/",
        )
        self.val_metrics = self.tra_metrics.clone(prefix="val/")
        self.val_metrics.add_metrics(
            {  # None
                # "Accuracy": torchmetrics.wrappers.ClasswiseWrapper(
                #     MulticlassAccuracy(
                #         num_classes=num_classes,
                #         average="none",
                #         ignore_index=ignore_index,
                #         # NOTE: Assign NaN to absent and ignored classes so that they can be
                #         # identified and excluded from the macroscopic averaging step.
                #         zero_division=torch.nan,
                #     ),
                #     prefix="Accuracy/",
                # ),
                "Precision": ClasswiseWrapper(
                    MulticlassPrecision(
                        num_classes=num_classes,
                        average="none",
                        ignore_index=ignore_index,
                        # NOTE: Assign NaN to absent and ignored classes so that they can be
                        # identified and excluded from the macroscopic averaging step.
                        zero_division=torch.nan,
                    ),
                    prefix="Precision/",
                ),
                "Recall": ClasswiseWrapper(
                    MulticlassRecall(
                        num_classes=num_classes,
                        average="none",
                        ignore_index=ignore_index,
                        # NOTE: Assign NaN to absent and ignored classes so that they can be
                        # identified and excluded from the macroscopic averaging step.
                        zero_division=torch.nan,
                    ),
                    prefix="Recall/",
                ),
                # "Specificity": torchmetrics.wrappers.ClasswiseWrapper(
                #     MulticlassSpecificity(
                #         num_classes=num_classes,
                #         average="none",
                #         ignore_index=ignore_index,
                #         # NOTE: Assign NaN to absent and ignored classes so that they can be
                #         # identified and excluded from the macroscopic averaging step.
                #         zero_division=torch.nan,
                #     ),
                #     prefix="Specificity/",
                # ),
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=num_classes,
                        average="none",
                        ignore_index=ignore_index,
                        # NOTE: Assign NaN to absent and ignored classes so that they can be
                        # identified and excluded from the macroscopic averaging step.
                        zero_division=torch.nan,
                    ),
                    prefix="IoU/",
                ),
            }
        )
        self.tst_metrics = self.val_metrics.clone(prefix="tst/")

        # Initialize the confusion matrix.
        self.tra_confmat = MulticlassConfusionMatrix(
            num_classes=num_classes, normalize="true"
        )
        self.val_confmat = self.tra_confmat.clone()
        self.tst_confmat = self.tra_confmat.clone()

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

    @override
    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        preds, target, loss = self._update_loss(batch)

        # TODO: Store step prefixes in an StrEnum.
        self.log("tra/" + "loss", loss)
        self.log_dict(self.tra_metrics(preds, target))

        self._plot_confmat(preds, target, step="tra")

        return loss

    def _plot_confmat(self, preds, target, step: Literal["tra", "val", "tst"]):
        tboard = self._get_tboard()
        if tboard is not None:
            confmat: MulticlassConfusionMatrix = getattr(self, f"{step}_confmat")
            confmat(preds, target)

            fig: Figure
            cax: Axes

            # NOTE: The color map must be set before plotting.
            cmap = plt.get_cmap("Blues")
            plt.set_cmap(cmap)

            fig, cax = confmat.plot()
            fig.set_size_inches(5, 5)

            # NOTE: The axis labels must be changed before searching for text artists
            # to ensure that it is version invariant.
            plt.xlabel("Predicted Class")
            plt.ylabel("True Class")

            # Change the value color to either black or white according to the
            # luminance of the underlying cell.
            vals: list[Text] = cax.findobj(
                lambda artist: isinstance(artist, Text)
                and artist.get_text() not in {"Predicted Class", "True Class"}
                and math.isclose(artist.get_rotation(), 0)
            )
            for val in vals:
                txt = val.get_text()
                val.set_color(get_fg_color(cmap(0 if txt == "" else float(txt))))

            # NOTE: The axis labels must be changed after searching for text artists
            # to ensure that it is version invariant.
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)

            plt.minorticks_off()

            tboard.add_figure(
                f"{step}/ConfusionMatrix", fig, global_step=self.global_step
            )

        # todo: is this needded? tboard automatically closes figs...
        plt.close()

    def _get_tboard(self) -> SummaryWriter | None:
        return (
            self.logger.experiment
            if hasattr(self.logger.experiment, "add_figure")
            else None
        )

    @override
    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        preds, target, loss = self._update_loss(batch)

        # TODO: Store step prefixes in an StrEnum.
        self.log("val/" + "loss", loss)
        self.log_dict(self.val_metrics(preds, target))

        self._plot_confmat(preds, target, step="val")

        # TODO: Clean up this block.
        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule
            batch["prediction"] = preds.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()

    @override
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        preds, target, loss = self._update_loss(batch)

        # TODO: Store step prefixes in an StrEnum.
        self.log("tst/" + "loss", loss)
        self.log_dict(self.tst_metrics(preds, target))

        self._plot_confmat(preds, target, step="tst")

    def _update_loss(self, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        input = batch["image"]
        target = batch["mask"]
        preds = self(input)
        loss = self.criterion(preds, target)

        return preds, target, loss
