import math
from collections.abc import Sequence
from typing import Any, Literal, Optional

import torch
import torchseg
from lightning import LightningModule
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
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassConfusionMatrix,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision,
                                         MulticlassRecall,
                                         MulticlassSpecificity, )
from typing_extensions import override

from training import model
from training.loss import CompoundLoss
from training.wrappers import MacroAverageWrapper
from utils.color import get_fg_color


class TrainingTask(LightningModule):
    #: Model to train.
    model: Any

    #: Performance metric to monitor in learning rate scheduler and callbacks.
    # TODO: See if renaming the loss messes up something.
    monitor = "val/loss"

    #: Whether the goal is to minimize or maximize the performance metric to monitor.
    mode = "min"

    # noinspection PyUnusedLocal
    def __init__(
        self,
        # The name of the model decoder.
        decoder: Literal[
            "unet",
            "unetplusplus",
            "manet",
            "linknet",
            "fpn",
            "pspnet",
            "pan",
            "deeplabv3",
            "deeplabv3plus",
        ],
        # The name of the model encoder.
        encoder: str,
        loss_params: dict[str, Any],
        encoder_weights: str | None = "imagenet",
        in_channels: int = 5,
        # TODO: See if we can remove background predictions from the model output.
        num_classes: int = 8 + 1,
        model_params: dict[
            str, dict[float | str | Sequence[float]] | float | str | Sequence[float]
        ]
        | None = None,
        ignore_index: Optional[int] = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        # The total number of warmup epochs.
        warmup_epochs: int = 15,
        # The total number of epochs constituting the period of the first cycle of the
        # annealing phase.
        T_0: int = 60,
        # The period ratio `Ti+1/Ti` of two consecutive cycles `i` of the annealing
        # phase.
        T_mult: int = 2,
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
    ):
        super().__init__()

        self.save_hyperparameters(
            # NOTE: See https://github.com/microsoft/torchgeo/pull/1897 for more
            # information.
            ignore=["ignore", "encoder_weights"]
            if encoder_weights is not None
            else ["ignore"]
        )

        self._init_model()
        self._loss = CompoundLoss(**loss_params)

        self._init_metrics()

    # TODO: Add support for freezing the encoder and or decoder in the supported
    #  architectures.
    def _init_model(self) -> None:
        encoder_weights: str | None
        try:
            encoder_weights = self.hparams.encoder_weights
        except AttributeError:
            # The encoder is pretrained.
            encoder_weights = None

        common_params = {
            "encoder_name": self.hparams.encoder,
            "encoder_weights": encoder_weights,
            "in_channels": self.hparams.in_channels,
            "classes": self.hparams.num_classes,
        }

        optional_params: (
            dict[
                str, dict[float | str | Sequence[float]] | float | str | Sequence[float]
            ]
            | None
        ) = self.hparams.model_params
        optional_params = optional_params if optional_params is not None else {}

        if self.hparams.decoder == "deeplabv3plus" and optional_params.get(
            "custom", False
        ):
            optional_params.pop("custom")
            self.model = model.DeepLabV3Plus(**common_params, **optional_params)
        else:
            optional_params.pop("attention", None)
            self.model = torchseg.create_model(
                self.hparams.decoder, **common_params, **optional_params
            )

    def _init_metrics(self) -> None:
        num_classes: int = self.hparams["num_classes"]

        base_params = {
            "num_classes": num_classes,
            "ignore_index": 0
            if self.hparams.loss_params["ignore_background"]
            else None,
            # NOTE: Assign NaN to absent and ignored classes so that they can be
            # identified and excluded from the macroscopic averaging step.
            "zero_division": torch.nan,
        }
        macro_params = base_params | {"average": "macro"}
        micro_params = base_params | {"average": "micro"}
        none_params = base_params | {"average": "none"}

        self.tra_metrics = MetricCollection(
            {
                # FIXME: Track macro accuracy and specificity.
                # Macro
                # "MacroAccuracy": MulticlassAccuracy(**macro_params),
                "MacroPrecision": MacroAverageWrapper(
                    MulticlassPrecision(**none_params)
                ),
                "MacroRecall": MacroAverageWrapper(MulticlassRecall(**none_params)),
                # "MacroSpecificity": MulticlassSpecificity(**macro_params),
                "MacroIoU": MacroAverageWrapper(MulticlassJaccardIndex(**none_params)),
                # Micro
                "MicroAccuracy": MulticlassAccuracy(**micro_params),
                "MicroPrecision": MulticlassPrecision(**micro_params),
                "MicroRecall": MulticlassRecall(**micro_params),
                "MicroSpecificity": MulticlassSpecificity(**micro_params),
                "MicroIoU": MulticlassJaccardIndex(**micro_params),
            },
            prefix="tra/",
        )
        self.val_metrics = self.tra_metrics.clone(prefix="val/")
        self.val_metrics.add_metrics(
            {  # None
                # "Accuracy": ClasswiseWrapper(
                #     MulticlassAccuracy(**none_params), prefix="Accuracy/"
                # ),
                "Precision": ClasswiseWrapper(
                    MulticlassPrecision(**none_params), prefix="Precision/"
                ),
                "Recall": ClasswiseWrapper(
                    MulticlassRecall(**none_params), prefix="Recall/"
                ),
                # "Specificity": ClasswiseWrapper(
                #     MulticlassSpecificity(**none_params), prefix="Specificity/"
                # ),
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(**none_params), prefix="IoU/"
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
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])

        warmup_epochs: int = self.hparams.warmup_epochs
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

    @override
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        y_hat: Tensor = self(x).softmax(dim=1)
        return y_hat

    def _update_loss(self, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        input = batch["image"]
        target = batch["mask"]
        preds = self(input)
        loss = self._loss(preds, target)

        return preds, target, loss

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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)
