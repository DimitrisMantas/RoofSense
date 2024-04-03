import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (EarlyStopping,
                                         ModelCheckpoint,
                                         GradientAccumulationScheduler,
                                         LearningRateMonitor, )
from lightning.pytorch.loggers import TensorBoardLogger

from classification.datamodules import TrainingDataModule
from classification.task import TrainingTask

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    task = TrainingTask(
        model="fcn",
        backbone="resnet18",
        weights=True,
        in_channels=6,
        num_classes=10,
        loss="jaccard",
        ignore_index=9,
    )

    datamodule = TrainingDataModule(
        root="../training/test", batch_size=64, patch_size=64, num_workers=8
    )

    # todo check strategies + callbacks + profiler
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath="logs/RoofSense",
                filename="best",
                monitor="val_loss",
                save_last=True,
            ),
            EarlyStopping(monitor="val_loss", patience=500),
            # TODO: LearningRateFinder(),
            GradientAccumulationScheduler(scheduling={0: 2}),
            LearningRateMonitor(),
            # TODO: OnExceptionCheckpoint(
            #     dirpath="logs/RoofSense",
            #     # Overwrite the last checkpoint.
            #     filename="last",
            # ),
            # TODO: lightning.pytorch.callbacks.SpikeDetection,
            # TODO: lightning.pytorch.callbacks.StochasticWeightAveraging
            # TODO: lightning.pytorch.callbacks.ModelPruning,
        ],
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir="logs/RoofSense"),
        benchmark=True,
        # TODO: profiler=AdvancedProfiler(dirpath="logs/RoofSense/profiling"),
        # detect_anomaly=True
        # fast_dev_run=True
    )

    trainer.fit(model=task, datamodule=datamodule)
