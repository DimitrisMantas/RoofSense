from __future__ import annotations

import os.path
from enum import StrEnum
from functools import cached_property
from types import ModuleType
from typing import Any, Final, Literal, Required, TypedDict, cast

import optuna
import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import ModuleDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchmetrics import ClasswiseWrapper, Metric, MetricCollection
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


# TODO: Support multi-task learning.
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
        self,
        # Loss
        loss_cfg: dict[str, Any],
        # The loss parameters.
        # Model
        encoder: str | None = None,
        # The model encoder.
        # This parameter is used to build an encoder-decoder model using SMP.
        # This parameter is ignored when a custom model is provided.
        decoder: str | None = None,
        # This parameter is used to build an encoder-decoder model using SMP.
        # This parameter is ignored when a custom model is provided.
        model: torch.nn.Module | None = None,
        # A custom model.
        # This feature is experimental!
        model_cfg: dict[str, Any] | None = None,
        # The model parameters.
        # This parameter is ignored when a custom model is provided.
        model_weights: str | None = "imagenet",
        # The pretrained model weights to use.
        # When using SAP to build the model:
        #   - A valid (https://smp.readthedocs.io/en/latest/encoders.html#choosing-the-right-encoder) weight name.
        #     The decoder is randomly initialized.
        #   - Any non-null string to load ImageNet-1K weights when the encoder is provided by TIMM.
        #     The decoder is randomly initialized.
        # Alternatively, pass a path to valid task checkpoint.
        # This parameter is ignored when a custom model is provided.
        freeze_encoder: bool = False,
        # True to freeze the encoder; False otherwise.
        # This parameter is ignored when a custom model is provided.
        freeze_decoder: bool = False,
        # True to freeze the encoder; False otherwise.
        # This parameter is ignored when a custom model is provided.
        # Optimizer
        optimizer: str | type[Optimizer] = "Adam",
        # The optimizer name.
        optimizer_cfg: dict[str, Any] = None,
        # The optimizer parameters.
        # LR Scheduler
        lr_scheduler: str | type[LRScheduler] = "PolynomialLR",
        # The learning rate scheduler name.
        lr_scheduler_cfg: dict[str, Any] = None,
        # The learning rate scheduler parameters.
        warmup_epochs: int = 0,
        # The length of the optional linear learning rate warmup period, measured in epochs.
        # TODO: Organize and document these parameters.
        # TODO: Check whether background predictions can be removed from the model output entirely.
        in_channels: int = 7,
        # The number of input channels to the model.
        # This parameter is ignored when a custom model is provided.
        num_classes: int = 8 + 1,
        # The number of output classes including the background.
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model" if model is not None else None)

        self.model = model if model is not None else None
        self.loss = self.configure_loss()
        self.metrics, self.confusn = self.configure_metrics()

    @cached_property
    def can_plot(self) -> bool:
        trainer = self.trainer
        return hasattr(trainer, "datamodule") and hasattr(trainer.datamodule, "plot")

    # TODO: Add type hints.
    @cached_property
    def experiment(self) -> Any | None:
        experiment = self.logger.experiment
        if hasattr(experiment, "add_figure"):
            # TensorBoard
            return experiment
        raise NotImplementedError(
            "Figure logging in only currently supported in TensorBoard."
        )

    @override
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        # Ensure the passed model is set to the task mode.
        model = self.model
        if model is not None:
            if model.training is not self.training:
                if self.training:
                    model.train()
                else:
                    model.eval()

    @override
    def configure_model(self) -> None:
        if self.model is not None:
            return

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
            # FIXME: This doesn't work.
            model.load_state_dict(
                load_model_from_lightning_checkpoint(self.hparams.model_weights)
            )

        if self.hparams.freeze_encoder:
            freeze_component(model, "encoder")
        if self.hparams.freeze_decoder:
            freeze_component(model, "decoder")

        self.model = model

    def configure_loss(self) -> torch.nn.modules.loss._Loss:
        return CompoundLoss(**self.hparams.loss_cfg)

    def configure_metrics(
        self,
    ) -> tuple[
        ModuleDict[TrainingStage, MetricCollection],
        ModuleDict[TrainingStage, MulticlassConfusionMatrix],
    ]:
        num_classes: int = self.hparams.num_classes
        ignore_index: int | None = (
            None if self.hparams.loss_cfg.get("include_background", True) else 0
        )

        def sanitize_and_prefix_metric_name(
            metric_class: type[Metric], prefix: str = ""
        ) -> str:
            return metric_class.__name__.replace("Multiclass", prefix)

        common_params = dict(num_classes=num_classes, ignore_index=ignore_index)
        micro_avg_params = common_params | {"average": "micro"}
        none_avg_params = common_params | {"average": "none"}

        metric_classes = [
            MulticlassAccuracy,
            MulticlassF1Score,
            MulticlassJaccardIndex,
            MulticlassPrecision,
            MulticlassRecall,
        ]
        metrics = ModuleDict(
            {
                TrainingStage.TRA: MetricCollection(
                    {
                        sanitize_and_prefix_metric_name(
                            metric_class, "Macro"
                        ): MacroAverageWrapper(
                            metric_class(**none_avg_params), ignore_index
                        )
                        for metric_class in metric_classes
                    }
                    | {
                        sanitize_and_prefix_metric_name(
                            metric_class, "Micro"
                        ): metric_class(**micro_avg_params)
                        for metric_class in metric_classes
                    },
                    prefix=f"{TrainingStage.TRA}/",
                )
            }
        )
        metrics[TrainingStage.VAL] = metrics[TrainingStage.TRA].clone(
            prefix=f"{TrainingStage.VAL}/"
        )
        metrics[TrainingStage.VAL].add_metrics(
            {
                sanitize_and_prefix_metric_name(metric_class): ClasswiseWrapper(
                    metric_class(**none_avg_params),
                    prefix=sanitize_and_prefix_metric_name(metric_class),
                )
                for metric_class in metric_classes
            }
        )
        metrics[TrainingStage.TST] = metrics[TrainingStage.VAL].clone(
            prefix=f"{TrainingStage.TST}/"
        )

        confmats = ModuleDict(
            {
                stage: MulticlassConfusionMatrix(num_classes, normalize="true")
                for stage in TrainingStage
            }
        )

        return metrics, confmats

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self._configure_optimizer()
        scheduler = self._configure_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor_train},
        }

    @override
    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    @override
    def training_step(self, batch: Batch) -> Tensor:
        pred, loss = self._step(batch, TrainingStage.TRA)

        self.log_dict(
            self.metrics[TrainingStage.TRA].compute(), on_step=True, on_epoch=False
        )
        if self.trainer._logger_connector.should_update_logs:
            self._compute_and_log_confmat(TrainingStage.TRA)

        return loss

    @override
    def on_train_epoch_end(self) -> None:
        self._reset_metrics(stage=TrainingStage.TRA)

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        pred, _ = self._step(batch, stage=TrainingStage.VAL)

        # TODO: Plot only when specified by the trainer.
        if (  # Plot the first 10 batches.
            batch_idx < 10 and self.can_plot
        ):
            batch["prediction"] = pred.argmax(dim=1)
            for key in batch.keys():
                batch[key] = batch[key].cpu()

            # Plot the first sample of each batch.
            sample = unbind_samples(batch)[0]
            try:
                fig: Figure = self.trainer.datamodule.plot(sample)
            except RGBBandsMissingError:
                return

            self._log_fig(f"Image/{batch_idx}", fig)

    @override
    def on_validation_epoch_end(self) -> None:
        self._on_eval_epoch_end(TrainingStage.VAL)

    @override
    def test_step(self, batch: Batch) -> None:
        self._step(batch, TrainingStage.TST)

    @override
    def on_test_epoch_end(self) -> None:
        self._on_eval_epoch_end(TrainingStage.TST)

    @override
    def predict_step(self, batch: Batch) -> Tensor:
        input = batch["image"]
        return self(input).softmax(dim=1)

    @staticmethod
    def _resolve_module_cls_and_cfg(
        module: str | type[Optimizer | LRScheduler],
        package: ModuleType,
        cfg: dict[str, Any] | None,
    ) -> tuple[type[Optimizer | LRScheduler], dict[str, Any]]:
        cls_ = getattr(package, module, module)
        cls_ = cast(cls_.__base__, cls_)

        cfg = cfg if cfg is not None else {}

        return cls_, cfg

    def _configure_optimizer(self) -> Optimizer:
        params: list[dict[str, Any]] | None = self.hparams.optimizer_cfg.get(
            "params", None
        )

        cls_, cfg = self._resolve_module_cls_and_cfg(
            self.hparams.optimizer, torch.optim, self.hparams.optimizer_cfg
        )
        # This is useful in cases when parameter groups are optionally specified by external functions which otherwise return None.
        cfg.pop("params", None)

        return cls_(self.parameters() if params is None else params, **cfg)

    def _configure_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        warmup_epochs: int = self.hparams.warmup_epochs

        cls_, cfg = self._resolve_module_cls_and_cfg(
            self.hparams.lr_scheduler,
            torch.optim.lr_scheduler,
            self.hparams.lr_scheduler_cfg,
        )

        annealing_scheduler = cls_(optimizer, **cfg)

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

    def _compute_and_log_loss(
        self, batch: Batch, stage: TrainingStage
    ) -> tuple[Tensor, Tensor, Tensor]:
        input: Tensor = batch["image"]
        target: Tensor = batch["mask"]

        pred: Tensor = self(input)

        loss: Tensor = self.loss(pred, target)
        self.log(f"{stage}/Loss", loss)

        return pred, target, loss

    def _log_fig(self, name: str, fig: Figure) -> None:
        # TODO: Support checking whether the logger exists before the figure is created.
        logger = self.experiment
        if logger is not None:
            logger.add_figure(name, fig, global_step=self.global_step)
        plt.close()

    def _compute_and_log_confmat(self, stage: TrainingStage) -> None:
        confmat, _ = self.confusn[stage].plot(
            self.confusn[stage].compute(), cmap="Blues"
        )
        self._log_fig(f"{stage}/ConfusionMatrix", confmat)

    def _reset_metrics(self, stage: TrainingStage) -> None:
        self.metrics[stage].reset()
        self.confusn[stage].reset()

    def _step(
        self, batch: Batch, stage: TrainingStage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred, target, loss = self._compute_and_log_loss(batch, stage)

        # Use manual logging.
        # See https://github.com/Lightning-AI/torchmetrics/issues/2683 for more information.
        self.metrics[stage].update(pred, target)
        self.confusn[stage].update(pred, target)

        return pred, loss

    def _on_eval_epoch_end(self, stage: TrainingStage) -> None:
        self.log_dict(self.metrics[stage].compute())
        self._compute_and_log_confmat(stage)

        self._reset_metrics(stage)


class Sample(TypedDict, total=False):
    image: Required[Tensor]
    mask: Required[Tensor]
    prediction: Tensor


class Batch(Sample):
    pass
