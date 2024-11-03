from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Final, Literal, cast, TypedDict, Required

import optuna
import torch
import torchseg  # type: ignore[import-untyped]
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from rasterio import CRS
from torch import Tensor
from torch.nn import Conv2d, Identity
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import BoundingBox, RGBBandsMissingError, unbind_samples
from torchmetrics import ClasswiseWrapper, Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchseg.base import SegmentationHead  # type: ignore[import-untyped]
from typing_extensions import override

from roofsense.metrics.classification.confusion_matrix import MulticlassConfusionMatrix
from roofsense.metrics.wrappers.macro_averaging import MacroAverageWrapper
from roofsense.training.loss import CompoundLoss1
from roofsense.utils.type import MetricKwargs


class TrainingTask(LightningModule):
    """Task used for training and performance evaluation purposes."""

    # TODO: See if renaming the loss messes up something.
    monitor = "val/loss"
    monitor_train: Final[str] = "val/loss"
    """The performance metric to monitor when using certain learning rate schedulers 
    (e.g., `ReduceLROnPlateau`) during training"""
    monitor_optim: Final[str] = "val/mIoU"
    """The performance metric to monitor when performing hyperparameter optimization."""
    monitor_train_direction: Final[Literal["max", "min"]] = "min"
    """The optimization direction of the training process with respect to the 
    corresponding performance metric to monitor."""
    monitor_optim_direction: Final[optuna.study.StudyDirection] = (
        optuna.study.StudyDirection.MAXIMIZE
    )
    """The optimization direction of the hyperparameter optimization process with 
    respect to the corresponding performance metric to monitor."""

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
        in_channels: int = 7,
        # TODO: See if we can remove background predictions from the model output.
        num_classes: int = 8 + 1,
        model_params: dict[
            str,
            dict[str, float | str | Sequence[float]] | float | str | Sequence[float],
        ]
        | None = None,
        ignore_index: int | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        # encoder_depth:int=5,
        # Optimizer
        optimizer: Literal["adam", "adamw"] = "adam",
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        # Match the Keras defaults.
        eps: float = 1e-7,
        weight_decay: float = 0,
        amsgrad: bool = False,
        # Scheduler
        # Warmup
        warmup_lr_pct: float = 1e-6,
        warmup_epochs: int = 0,
        # Annealing
        annealing: Literal["cos", "poly"] = "poly",
        poly_decay_exp: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.init_model()
        self._loss = CompoundLoss1(**loss_params)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.init_metrics()

    def init_model(self) -> None:
        """Configure the underlying model."""
        encoder = self.hparams.encoder
        in_channels = self.hparams.in_channels
        num_classes = self.hparams.num_classes

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
            "encoder_name": encoder,  # "encoder_depth": self.hparams.encoder_depth,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": num_classes,
        }

        optional_params: (
            dict[
                str, dict[float | str | Sequence[float]] | float | str | Sequence[float]
            ]
            | None
        ) = self.hparams.model_params
        optional_params = optional_params if optional_params is not None else {}

        # NOTE: This needs to be here because otherwise the model is not initialized.
        self.model = torchseg.create_model(**common_params, **optional_params)

        # Replace weights with ones passed if passed.
        encoder_weights = self.hparams.encoder_weights
        if encoder_weights is not None and encoder_weights != "imagenet":
            # Load the pretrained weights.
            ckpt = torch.load(encoder_weights)

            weights: OrderedDict[str, Tensor] = ckpt["state_dict"]
            weights = OrderedDict(
                {name: value for name, value in weights.items() if "model." in name}
            )
            weights = OrderedDict(
                {
                    name.replace("model.", "", 1): value
                    for name, value in weights.items()
                }
            )

            # Prepare to accept the weights.
            temp_common_params = common_params.copy()
            # TODO: Infer this parameter automatically.
            temp_common_params["in_channels"] = weights[
                "encoder.model.conv1.0.weight"
            ].size(dim=1)
            temp_common_params["classes"] = weights["segmentation_head.0.weight"].size(
                dim=0
            )

            self.model = torchseg.create_model(**temp_common_params, **optional_params)

            if "aux_params" in optional_params:
                # Copy the classifier weights to the pretrained state dictionary.
                weights["classification_head.3.weight"] = (
                    self.model.classification_head[3].weight.data.clone()
                )
                weights["classification_head.3.bias"] = self.model.classification_head[
                    3
                ].bias.data.clone()

            # Push the weights to the model.
            self.model.load_state_dict(weights)

            # Configure the model input and output to match the current task.
            # Replace the entry point.
            conv1_name: str = self.model.encoder.model.pretrained_cfg["first_conv"]
            # strip extra info in case conv1 is a list
            conv1_name = conv1_name.split(".", maxsplit=1)[0]
            old_conv1: Conv2d = getattr(self.model.encoder.model, conv1_name)
            con1_is_seq = False
            if isinstance(old_conv1, torch.nn.modules.container.Sequential):
                # conv1 is a list
                con1_is_seq = True
                old_conv1 = old_conv1[0]
            new_conv1 = Conv2d(
                in_channels=in_channels,
                out_channels=old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                dilation=old_conv1.dilation,
                groups=old_conv1.groups,
                bias=old_conv1.bias,
                padding_mode=old_conv1.padding_mode,
            )

            new_conv1.weight.data[:, : old_conv1.weight.shape[1], ...] = (
                old_conv1.weight.data.clone()
            )
            if con1_is_seq:
                # conv1 is a list
                self.model.encoder.model.conv1[0] = new_conv1
            else:
                self.model.encoder.model.conv1 = new_conv1

            # if "aux_params" in optional_params:
            # Replace the segmentation head.
            old_head: SegmentationHead = self.model.segmentation_head
            new_head = SegmentationHead(
                in_channels=self.model.decoder.out_channels,
                out_channels=num_classes,
                activation=Identity(),
                kernel_size=1,
                upsampling=4,
            )

            self.model.segmentation_head = new_head

        if self.hparams["freeze_backbone"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        if self.hparams["freeze_decoder"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def _freeze_component(self, name: Literal["encoder", "decoder"]) -> None:
        component: torch.nn.Module = getattr(self.model, name)
        for param in component.parameters():
            param.requires_grad = False

    def init_metrics(self) -> None:
        """Initialize the performance metrics for each stage."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = (
            None if self.hparams.loss_params.get("include_background", True) else 0
        )

        common_params: MetricKwargs = {
            "num_classes": num_classes,
            "ignore_index": ignore_index,
        }
        micro_params: MetricKwargs = common_params | {"average": "micro"}
        base_none_params: MetricKwargs = common_params | {"average": "none"}
        none_params_div_zero: MetricKwargs = base_none_params | {
            # NOTE: Some metrics accept an argument to handle division-by-zero cases
            # when absent or ignored classes are encountered. See
            # https://github.com/Lightning-AI/torchmetrics/pull/2198 for more
            # information.
            "zero_division": 0
        }
        none_params_div_nan: MetricKwargs = base_none_params | {
            # NOTE: The valid value range of this parameter varies by metric.
            "zero_division": torch.nan
        }

        self.tra_metrics = MetricCollection(
            {  # Macroscopic Metrics
                # NOTE: This metric may not account for samplewise absent classes
                # correctly but is still useful as a lower bound for its actual
                # value. See https://github.com/Lightning-AI/torchmetrics/pull/2443
                # for more information.
                "MacroAccuracy":  # NOTE: This wrapper improves the lower bound by properly ignoring
                # the background class at the reduction step.
                MacroAverageWrapper(
                    MulticlassAccuracy(**base_none_params), ignore_index=ignore_index
                ),
                "MacroPrecision":  # NOTE: This wrapper fixes a bug where the macroscopically reduced
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
                ),  # Microscopic Metrics
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
                "F1Score": ClasswiseWrapper(
                    MulticlassF1Score(**none_params_div_zero), prefix="F1Score/"
                ),
                "IoU": ClasswiseWrapper(
                    MulticlassJaccardIndex(**none_params_div_nan), prefix="IoU/"
                ),
            }
        )
        self.tst_metrics = self.val_metrics.clone(prefix="tst/")

        self._configure_confmat()

    def _configure_confmat(self) -> None:
        num_classes: int = self.hparams["num_classes"]

        for stage in TrainingStage:
            setattr(
                self,
                f"{stage}_confmat",
                MulticlassConfusionMatrix(num_classes, normalize="true"),
            )

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self._configure_optimizer()
        scheduler = self._configure_lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def _configure_optimizer(self) -> Optimizer:
        optimizer: str = self.hparams["optimizer"]

        if optimizer == "adam":
            cls = Adam
        elif optimizer == "adamw":
            cls = AdamW
        else:
            raise ValueError(
                f"Expected either 'adam' or 'adamw' as value of 'optimizer', but got {optimizer}."
            )

        return cls(
            self.parameters(),
            self.hparams["lr"],
            self.hparams["betas"],
            self.hparams["eps"],
            self.hparams["weight_decay"],
            self.hparams["amsgrad"],
        )

    def _configure_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        max_epochs: int = self.trainer.max_epochs
        # TODO: Try to infer the epoch limit from 'max_steps' first if it has been specified.
        max_epochs = max_epochs if max_epochs > 1 else 1000

        warmup_epochs: int = self.hparams["warmup_epochs"]
        if warmup_epochs > max_epochs:
            raise ValueError(
                f"Specified warmup length ({warmup_epochs} epochs) cannot be larger than training duration {max_epochs} epochs)."
            )

        annealing_scheduler = self._configure_lr_annealing_scheduler(
            max_epochs, optimizer, warmup_epochs
        )

        if warmup_epochs == 0:
            return annealing_scheduler
        else:
            return SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(
                        optimizer,
                        start_factor=self.hparams["warmup_lr_pct"],
                        total_iters=warmup_epochs,
                    ),
                    annealing_scheduler,
                ],
                milestones=[warmup_epochs],
            )

    def _configure_lr_annealing_scheduler(
        self, max_epochs: int, optimizer: Optimizer, warmup_epochs: int
    ) -> LRScheduler:
        annealing: str = self.hparams["annealing"]

        scheduler: CosineAnnealingLR | PolynomialLR
        if annealing == "cos":
            scheduler = cast(
                CosineAnnealingLR,
                CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs),
            )
        elif annealing == "poly":
            scheduler = cast(
                PolynomialLR,
                PolynomialLR(
                    optimizer,
                    total_iters=max_epochs - warmup_epochs,
                    power=self.hparams["poly_decay_exp"],
                ),
            )
        else:
            raise ValueError(
                f"Expected either 'cos' or 'poly' as value of 'annealing', but got {annealing}."
            )
        return scheduler

    @override
    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        mask, label, target, loss = self._update_loss(batch)

        self.log(f"{TrainingStage.TRA}/loss", loss)

        # `MetricCollection` instances need to be logged manually.
        # See https://github.com/Lightning-AI/torchmetrics/issues/2683 for more information.
        self.tra_metrics.update(mask, target)
        self.log_dict(
            self.tra_metrics.compute(),
            # TODO: Check whether this is needed.
            on_step=True,
            on_epoch=False,
        )

        self.tra_confmat.update(mask, target)
        confmat = self.tra_confmat.compute()
        if self._should_log():
            tboard = self._get_tensorboard()
            if tboard is not None:
                fig, _ = self.tra_confmat.plot(confmat, cmap="Blues")
                tboard.add_figure(
                    "tra/ConfusionMatrix", fig, global_step=self.global_step
                )
                plt.close()

        return loss

    @override
    def on_train_epoch_end(self) -> None:
        for suffix in ["metrics", "confmat"]:
            metric: Metric = getattr(self, f"{TrainingStage.TRA}_{suffix}")
            metric.reset()

    def _should_log(self) -> bool:
        logging_freq: int = self.trainer.log_every_n_steps  # type: ignore[attr-defined]
        # Start counting from 1 to match Lightning.
        global_step = self.global_step + 1

        return (global_step >= logging_freq) and (global_step % logging_freq) == 0

    @override
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        # self._plot_confmat(
        #     step="val"
        # )
        tboard = self._get_tensorboard()
        if tboard is not None:
            fig, _ = self.val_confmat.plot(cmap="Blues")
            tboard.add_figure("val/ConfusionMatrix", fig, global_step=self.global_step)
            plt.close()
        self.val_confmat.reset()

    @override
    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        mask, label, target, loss = self._update_loss(batch)

        # TODO: Store step prefixes in an StrEnum.
        self.log("val/" + "loss", loss)

        # # https: // lightning.ai / docs / torchmetrics / stable / pages / lightning.html  # common-pitfalls
        # self.val_metrics(mask, target)
        # self.log_dict(self.val_metrics)

        # https://github.com/Lightning-AI/torchmetrics/issues/2683
        self.val_metrics.update(mask, target)
        self.val_confmat.update(mask, target)

        # self._plot_confmat(mask, target, step="val")

        # TODO: Clean up this block.
        # TODO: Plot only when specified by the trainer.
        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule
            batch["prediction"] = mask.argmax(dim=1)
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
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.tst_metrics.compute())
        self.tst_metrics.reset()

        # self._plot_confmat(
        #     step="tst"
        # )
        tboard = self._get_tensorboard()
        if tboard is not None:
            fig, _ = self.tst_confmat.plot(cmap="Blues")
            tboard.add_figure("tst/ConfusionMatrix", fig, global_step=self.global_step)
            plt.close()
        self.tst_confmat.reset()

    @override
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        mask, label, target, loss = self._update_loss(batch)

        # TODO: Store step prefixes in an StrEnum.
        self.log("tst/" + "loss", loss)

        # # https: // lightning.ai / docs / torchmetrics / stable / pages / lightning.html  # common-pitfalls
        # self.tra_metrics(mask, target)
        # self.log_dict(self.tra_metrics)

        # https://github.com/Lightning-AI/torchmetrics/issues/2683
        self.tst_metrics.update(mask, target)
        self.tst_confmat.update(mask, target)

        # self._plot_confmat(mask, target, step="tst")

    @override
    def predict_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        image = batch["image"]
        probs = self.forward(image).softmax(dim=1)
        return probs

    def _update_loss(self, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        input: Tensor = batch["image"]
        target: Tensor = batch["mask"]

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

        loss = self._loss(mask, target)

        return mask, label, target, loss + 0.4 * aux_loss

    def _get_tensorboard(self) -> SummaryWriter | None:
        # TODO: Type hint this variable.
        experiment = self.logger.experiment  # type: ignore[attr-defined]
        return (
            experiment
            # TODO: A more complete check would look at the type of experiment.
            if hasattr(experiment, "add_figure")
            else None
        )

    @override
    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)


class Sample(TypedDict, total=False):
    image: Required[Tensor]
    mask: Required[Tensor]
    # GeoDataset
    crs: CRS
    bounds: BoundingBox
    # Model -> Plot
    prediction: Tensor


class Batch(Sample):
    pass


class TrainingStage(StrEnum):
    TRA = "tra"
    VAL = "val"
    TST = "tst"
