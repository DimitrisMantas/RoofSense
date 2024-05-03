import lightning.pytorch
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (EarlyStopping,
                                         ModelCheckpoint,
                                         LearningRateMonitor, )
from lightning.pytorch.loggers import TensorBoardLogger

from training.datamodule import TrainingDataModule
from training.task import TrainingTask

# from training.task import TrainingTask, PerformanceMetricAverage, PerformanceMetric

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(42, workers=True)

    task = TrainingTask(
        model="unet",
        backbone="resnet18",
        weights=True,
        in_channels=5,
        num_classes=8+1,
        loss="ce",
        class_weights=torch.tensor(np.load("../dataset/temp/weights.npy"),dtype=torch.float32),
        ignore_index=0,
    )

    datamodule = TrainingDataModule(
        # TODO: Try a batch size of 12.
        root="../dataset/temp", batch_size=16, num_workers=8
    )

    # todo check strategies + callbacks + profiler
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath="../logs/RoofSense",
                filename="best",
                monitor="val_loss",
                save_last=True,
            ),
            EarlyStopping(monitor="val_loss", patience=500),
            # TODO: LearningRateFinder(),
            # GradientAccumulationScheduler(scheduling={0: 3}),
            LearningRateMonitor(),  # TODO: OnExceptionCheckpoint(
            #     dirpath="logs/RoofSense",
            #     # Overwrite the last checkpoint.
            #     filename="last",
            # ),
            # TODO: lightning.pytorch.callbacks.SpikeDetection,
            # TODO: lightning.pytorch.callbacks.StochasticWeightAveraging
            # TODO: lightning.pytorch.callbacks.ModelPruning,
        ],
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir="../logs/RoofSense"),
        benchmark=True,
        # TODO: profiler=AdvancedProfiler(dirpath="logs/RoofSense/profiling"),
        # detect_anomaly=True
        # fast_dev_run=True
        # overfit_batches=1
    )

    trainer.fit(model=task, datamodule=datamodule)
