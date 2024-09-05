from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Final, Literal

import optuna
import torch
import torchseg  # type: ignore[import-untyped]
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import Conv2d, Identity
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      LinearLR,
                                      LRScheduler,
                                      PolynomialLR,
                                      SequentialLR, )
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassF1Score,
                                         MulticlassJaccardIndex,
                                         MulticlassPrecision,
                                         MulticlassRecall, )
from torchseg.base import SegmentationHead  # type: ignore[import-untyped]
from typing_extensions import override

from metrics.classification.confusion_matrix import MulticlassConfusionMatrix
from metrics.wrappers.macro_averaging import MacroAverageWrapper
from training.loss import CompoundLoss1
from utils.type import MetricKwargs


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
        encoder_weights: str | Literal["imagenet"] | None = "imagenet",
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

        # TODO: Should we be doing this?
        # self.encoder_weights=encoder_weights
        self.save_hyperparameters(  # ignore=encoder_weights
        )

        self.init_model()
        self._loss = CompoundLoss(**loss_params)
        self._init_metrics()

    def init_model(self) -> None:
        """Configure the underlying model."""
        encoder = self.hparams.encoder
        in_channels = self.hparams.in_channels

        # --------------------------------------------------------------------------------------------

        # TODO: Ideally, we should init with imagenet, save the weights somewhere
        #            before loading in the new ones, and then load them back in to
        #           replace the random inits introduced by the loading process.

        # if the weights is a string then a checkpoint path was passed. do random
        # init and then replace the weights with the ones passed.
        encoder_weights = self.hparams.encoder_weights
        if encoder_weights is not None and encoder_weights != "imagenet":
            # a checkpoint was passed.
            encoder_weights = None
        # --------------------------------------------------------------------------------------------

        common_params = {
            "arch": self.hparams.decoder,
            "encoder_name": encoder,
            # "encoder_depth": self.hparams.encoder_depth,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": self.hparams.num_classes,
        }

        optional_params: (
            dict[
                str, dict[float | str | Sequence[float]] | float | str | Sequence[float]
            ]
            | None
        ) = self.hparams.model_params
        optional_params = optional_params if optional_params is not None else {}

        self.model = torchseg.create_model(**common_params, **optional_params)

        # Replace weights with ones passed if passed.
        encoder_weights = self.hparams.encoder_weights
        if encoder_weights is not None and encoder_weights != "imagenet":
            # load the weights
            encoder_name, state_dict = get_encoder_params(encoder_weights)
            if encoder_name is not None and encoder_name != encoder:
                msg = f"Encoder weights: {encoder_weights!r} for encoder: {encoder_name!r} incompatible with specified encoder: {encoder!r}."
                raise ValueError(msg)
            # prepare model to accept the weights
            # todo:infer the name of the first layer from the encoder.
            self.model.encoder.model.conv1 = reinit_initial_conv_layer(
                # todo: infer the original input channels from the weights
                self.model.encoder.model.conv1,
                new_in_channels=4,
                keep_first_n_weights=None,
            )
            # push the weights to the model
            self.model.encoder.load_state_dict(state_dict)
            # adjust model to the specified number of input channels
            self.model.encoder.model.conv1 = reinit_initial_conv_layer(
                self.model.encoder.model.conv1,
                new_in_channels=in_channels,
                keep_first_n_weights=4,
            )

        if self.hparams["freeze_backbone"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        if self.hparams["freeze_decoder"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def _init_metrics(self) -> None:
        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = (
            0 if self.hparams.loss_params["ignore_background"] else None
        )

        common_params = {"num_classes": num_classes, "ignore_index": ignore_index}
        micro_params = common_params | {"average": "micro"}
        base_none_params = common_params | {"average": "none"}
        none_params_div_zero = base_none_params | {
            # NOTE: Some metrics accept an argument to handle division-by-zero cases
            # when absent or ignored classes are encountered. See
            # https://github.com/Lightning-AI/torchmetrics/pull/2198 for more
            # information.
            "zero_division": 0
        }
        none_params_div_nan = base_none_params | {
            # NOTE: The valid value range of this parameter varies by metric.
            "zero_division": torch.nan
        }

        self.tra_metrics = MetricCollection(
            {  # Macroscopic Metrics
                # NOTE: This metric may not account for samplewise absent classes
                # correctly but is still useful as a lower bound for its actual
                # value. See https://github.com/Lightning-AI/torchmetrics/pull/2443
                # for more information.
                "MacroAccuracy":
                # NOTE: This wrapper improves the lower bound by properly ignoring
                # the background class at the reduction step.
                MacroAverageWrapper(
                    MulticlassAccuracy(**base_none_params), ignore_index=ignore_index
                ),
                "MacroPrecision":
                # NOTE: This wrapper fixes a bug where the macroscopically reduced
                # value of the underlying metric would be NaN if one or more
                # class-wise values where not defined. See
                # https://github.com/Lightning-AI/torchmetrics/issues/2535 for more
                # information.
                MacroAverageWrapper(
                    MulticlassPrecision(**none_params_div_zero),
                    ignore_index=ignore_index,
                ),
                "MacroRecall": MacroAverageWrapper(
                    MulticlassRecall(**none_params_div_zero), ignore_index=ignore_index
                ),
                "MacroF1Score": MacroAverageWrapper(
                    MulticlassF1Score(**none_params_div_zero), ignore_index=ignore_index
                ),
                "MacroIoU": MacroAverageWrapper(
                    MulticlassJaccardIndex(**none_params_div_nan),
                    ignore_index=ignore_index,
                ),
                # Microscopic Metrics
                "MicroAccuracy": MulticlassAccuracy(**micro_params),
                "MicroIoU": MulticlassJaccardIndex(**micro_params),
            },
            prefix="tra/",
        )
        self.val_metrics = self.tra_metrics.clone(prefix="val/")
        self.val_metrics.add_metrics(
            {  # Classwise Metrics
                "Accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(**base_none_params), prefix="Accuracy/"
                ),
                "Precision": ClasswiseWrapper(
                    MulticlassPrecision(**none_params_div_zero), prefix="Precision/"
                ),
                "Recall": ClasswiseWrapper(
                    MulticlassRecall(**none_params_div_zero), prefix="Recall/"
                ),
                "F1Score": ClasswiseWrapper(
                    MulticlassF1Score(**none_params_div_zero), prefix="F1Score/"
                ),
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(**none_params_div_nan), prefix="IoU/"
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

        model_params = self.hparams.model_params
        model_params = model_params if model_params is not None else {}
        if "aux_params" in model_params:
            # The auxiliary classification head is active.
            mask, label = preds

            # Build the target label tensor.
            label_target = torch.zeros(
                (label.shape[0], self.hparams.num_classes),
                dtype=label.dtype,
                device=label.device,
            )
            item: Tensor
            for i, item in enumerate(target.view(target.shape[0], -1)):
                label_target[i, item.unique()] = 1

            # Compute the auxiliary loss.
            aux_loss = self.cls_loss(label, label_target)[:, 1:] * self._loss.weight[1:]
            aux_loss = aux_loss.sum(dim=1).mean()
        else:
            mask, label = preds, None
            aux_loss = 0



        return mask, label, target, loss + 0.4 * aux_loss


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
