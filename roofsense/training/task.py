from __future__ import annotations

import os.path
from enum import StrEnum
from typing import Any, Final, Literal, Required, TypedDict

import optuna
import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)
from typing_extensions import override

from roofsense.metrics.classification.confusion_matrix import MulticlassConfusionMatrix
from roofsense.metrics.wrappers.macro_averaging import MacroAverageWrapper
from roofsense.training.loss import CompoundLoss
from roofsense.utilities.model import (
    freeze_component,
    load_model_from_lightning_checkpoint,
)


class TrainingStage(StrEnum):
    TRA = "tra"
    VAL = "val"
    TST = "tst"


# TODO: Support custom models.
# TODO: Support multi-task learning.
# TODO: Support optimizer parameter groups.
class TrainingTask(LightningModule):
    """Task used for training and performance evaluation purposes."""

    monitor_train: Final[str] = f"{TrainingStage.VAL}/Loss"
    """The performance metric to monitor when using certain learning rate schedulers (e.g., `ReduceLROnPlateau`) during training."""
    monitor_optim: Final[str] = f"{TrainingStage.VAL}/MacroJaccardIndex"
    """The performance metric to monitor when performing hyperparameter optimization."""
    monitor_train_direction: Final[Literal["max", "min"]] = "min"
    """The optimization direction of the training process with respect to the corresponding performance metric to monitor."""
    monitor_optim_direction: Final[optuna.study.StudyDirection] = (
        optuna.study.StudyDirection.MAXIMIZE
    )
    """The optimization direction of the hyperparameter optimization process with respect to the corresponding performance metric to monitor."""

    def __init__(
        self,  # Model
        encoder: str,  # The encoder of the model.
        decoder: str,  # Loss
        loss_cfg: dict[str, Any],  # Model
        # The decoder of the model.
        model_cfg: dict[str, Any]
        | None = None,  # The arguments of the encoder and decoder factories.
        model_weights: str | None = "imagenet",  # The pretrained model weights to use.
        # A valid (https://smp.readthedocs.io/en/latest/encoders.html#choosing-the-right-encoder) weight name when using SMP encoders or any non-null string to load ImageNet-1K weights when using TIMM encoders.
        # The decoder is randomly initialized in either case.
        # Alternatively, pass a path to valid task checkpoint or None to randomly initialize the whole model.
        freeze_encoder: bool = False,  # True to freeze the encoder; False otherwise.
        freeze_decoder: bool = False,  # True to freeze the decoder; False otherwise.
        # Optimizer
        optimizer: str | type[Optimizer] = "Adam",  # The optimizer to use.
        optimizer_cfg: dict[str, Any] = None,  # The arguments of the optimizer factory.
        # LR Scheduler
        scheduler: str
        | type[LRScheduler] = "PolynomialLR",  # The learning rate scheduler to use.
        scheduler_cfg: dict[str, Any] = None,  # The arguments of the scheduler factory.
        warmup_epochs: int = 0,  # The length of the optional linear learning rate warmup period in epochs.
        # Metrics
        in_channels: int = 7,  # TODO: See if we can remove background predictions from the model output.
        num_classes: int = 8 + 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.configure_model()
        self.loss = self.configure_loss()
        self.configure_metrics()

    def configure_model(self) -> torch.nn.Module:
        model_weights: str | None = self.hparams.model_weights
        if os.path.isfile(model_weights):
            # A model checkpoint was passed.
            model_weights = None

        common_params = dict(
            arch=self.hparams.decoder,
            encoder_name=self.hparams.encoder,
            encoder_weights=model_weights,
            in_channels=self.hparams.in_channels,
            classes=self.hparams.num_classes,
        )
        optional_params: dict[str, Any] | None = self.hparams.model_cfg
        optional_params = optional_params if optional_params is not None else {}

        model = smp.create_model(**common_params, **optional_params)
        if model_weights is not self.hparams.model_weights:
            # A model checkpoint was passed.
            model.load_state_dict(
                load_model_from_lightning_checkpoint(self.hparams.model_weights)
            )

        if self.hparams.freeze_encoder:
            freeze_component(self.model, "encoder")
        if self.hparams.freeze_decoder:
            freeze_component(self.model, "decoder")

        return model

    def configure_loss(self) -> torch.nn.modules.loss._Loss:
        return CompoundLoss(**self.hparams.loss_cfg)

    # TODO: Clean up.
    def configure_metrics(self) -> None:
        num_classes: int = self.hparams.num_classes
        ignore_index: int | None = (
            None if self.hparams.loss_cfg.get("include_background", True) else 0
        )

        common_params = dict(num_classes=num_classes, ignore_index=ignore_index)
        macro_avg = common_params | {"average": None}
        micro_avg = common_params | {"average": "micro"}

        metric_classes = [
            MulticlassAccuracy,
            MulticlassF1Score,
            MulticlassJaccardIndex,
            MulticlassPrecision,
            MulticlassRecall,
        ]
        for stage in TrainingStage:
            macro_metrics = {
                metric_class.__name__.replace(
                    "Multiclass", "Macro"
                ): MacroAverageWrapper(metric_class(**macro_avg), ignore_index)
                for metric_class in metric_classes
            }
            micro_metrics = {
                metric_class.__name__.replace("Multiclass", "Micro"): metric_class(
                    **micro_avg
                )
                for metric_class in metric_classes
            }
            metrics = macro_metrics | micro_metrics
            if stage in [TrainingStage.VAL, TrainingStage.TST]:
                class_metrics = {}
                for metric_class in metric_classes:
                    metric_label = metric_class.__name__.replace("Multiclass", "")
                    class_metrics[metric_label] = ClasswiseWrapper(
                        metric_class(**macro_avg), prefix=metric_label
                    )
                metrics |= class_metrics
            setattr(
                self, f"{stage}_metrics", MetricCollection(metrics, prefix=f"{stage}/")
            )
            setattr(
                self,
                f"{stage}_confmat",
                MulticlassConfusionMatrix(num_classes, normalize="true"),
            )

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self._configure_optimizer()
        scheduler = self._configure_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor_train},
        }

    def _configure_optimizer(self) -> Optimizer:
        optimizer_cls: type[torch.optim.optimizer.Optimizer] = getattr(
            torch.optim, self.hparams.optimizer, self.hparams.optimizer
        )
        optimizer_cfg: dict[str, Any] = self.hparams.optimizer_cfg
        if optimizer_cfg is None:
            optimizer_cfg = {}
        return optimizer_cls(self.parameters(), **optimizer_cfg)

    def _configure_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        warmup_epochs: int = self.hparams.warmup_epochs

        scheduler_cls: type[LRScheduler] = getattr(
            torch.optim.lr_scheduler, self.hparams.scheduler, self.hparams.scheduler
        )
        scheduler_cfg: dict[str, Any] = self.hparams.scheduler_cfg
        if scheduler_cfg is None:
            scheduler_cfg = {}
        annealing_scheduler = scheduler_cls(optimizer, **scheduler_cfg)

        if warmup_epochs == 0:
            return annealing_scheduler

        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-6, total_iters=warmup_epochs
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, annealing_scheduler],
            milestones=[warmup_epochs],
        )

    @override
    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    @override
    def training_step(self, batch: Batch) -> Tensor:
        pred, target, loss = self._compute_and_log_loss(batch, stage=TrainingStage.TRA)

        # Use manual logging.
        # See https://github.com/Lightning-AI/torchmetrics/issues/2683 for more information.
        self.tra_metrics.update(pred, target)
        self.log_dict(self.tra_metrics.compute(), on_step=True, on_epoch=False)

        self.tra_confmat.update(pred, target)
        if self.trainer._logger_connector.should_update_logs:
            logger = self._get_logger()
            if logger is not None:
                fig, _ = self.tra_confmat.plot(self.tra_confmat.compute(), cmap="Blues")
                logger.add_figure(
                    f"{TrainingStage.TRA}/ConfusionMatrix",
                    fig,
                    global_step=self.global_step,
                )
                plt.close()

        return loss

    @override
    def on_train_epoch_end(self) -> None:
        self.tra_metrics.reset()
        self.tra_confmat.reset()

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        pred, target, loss = self._compute_and_log_loss(batch, stage=TrainingStage.VAL)

        # Use manual logging.
        # See https://github.com/Lightning-AI/torchmetrics/issues/2683 for more information.
        self.val_metrics.update(pred, target)
        self.val_confmat.update(pred, target)

        # TODO: Clean up this block.
        # TODO: Plot only when specified by the trainer.
        if (
            # self.trainer._logger_connector.should_update_logs
            # and
            # Plot the first 10 batches.
            batch_idx < 10 and self._has_plotter()
        ):
            batch["prediction"] = pred.argmax(dim=1)
            for key in batch.keys():
                batch[key] = batch[key].cpu()
            # Plot the first sample of the first 10 batches.
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = self.trainer.datamodule.plot(sample)
            except RGBBandsMissingError:
                pass
            if fig is not None:
                logger = self._get_logger()
                if logger is not None:
                    logger.add_figure(
                        f"Image/{batch_idx}", fig, global_step=self.global_step
                    )
                plt.close()

    @override
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        logger = self._get_logger()
        if logger is not None:
            fig, _ = self.val_confmat.plot(self.val_confmat.compute(), cmap="Blues")
            logger.add_figure(
                f"{TrainingStage.VAL}/ConfusionMatrix",
                fig,
                global_step=self.global_step,
            )
            plt.close()
        self.val_confmat.reset()

    @override
    def test_step(self, batch: Batch) -> None:
        pred, target, loss = self._compute_and_log_loss(batch, stage=TrainingStage.TST)

        self.tst_metrics.update(pred, target)
        self.tst_confmat.update(pred, target)

    @override
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.tst_metrics.compute())
        self.tst_metrics.reset()

        logger = self._get_logger()
        if logger is not None:
            fig, _ = self.tst_confmat.plot(self.tst_confmat.compute(), cmap="Blues")
            logger.add_figure(
                f"{TrainingStage.TST}/ConfusionMatrix",
                fig,
                global_step=self.global_step,
            )
            plt.close()
        self.tst_confmat.reset()

    @override
    def predict_step(self, batch: Batch) -> Tensor:
        input = batch["image"]
        return self(input).softmax(dim=1)

    def _compute_and_log_loss(
        self, batch: Batch, stage: TrainingStage
    ) -> tuple[Tensor, Tensor, Tensor]:
        input: Tensor = batch["image"]
        target: Tensor = batch["mask"]

        pred: Tensor = self(input)

        loss: Tensor = self.loss(pred, target)
        self.log(f"{stage}/Loss", loss)

        return pred, target, loss

    # TODO: Add type hints.
    def _get_logger(self) -> Any | None:
        experiment = self.logger.experiment
        if hasattr(experiment, "add_figure"):
            # TensorBoard
            return experiment
        raise NotImplementedError(
            "Figure logging in only currently supported in TensorBoard."
        )

    def _has_plotter(self) -> bool:
        return hasattr(self.trainer, "datamodule") and hasattr(
            self.trainer.datamodule, "plot"
        )


class Sample(TypedDict, total=False):
    image: Required[Tensor]
    mask: Required[Tensor]
    prediction: Tensor


class Batch(Sample):
    pass
