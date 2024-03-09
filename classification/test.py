import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler

from classification.datamodules import TrainingDataModule
from classification.task import TrainingTask

if __name__ == "__main__":
    # datamodule=LandCoverAIDataModule(root="data/LandCoverAI",batch_size=8,num_workers=8,persistent_workers=True,pin_memory=True)
    datamodule = TrainingDataModule(
        root="../training/test",
        batch_size=64,
        patch_size=64,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="logs/RoofSense", save_top_k=10, save_last=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=50
    )

    batch_size_callback = lightning.pytorch.callbacks.BatchSizeFinder()

    logger = TensorBoardLogger(save_dir="logs/RoofSense")
    task = TrainingTask(
        model="fcn",
        backbone="resnet18",
        weights=True,
        in_channels=6,
        num_classes=10,
        loss="jaccard",
        ignore_index=9,
    )

    profiler = AdvancedProfiler(dirpath="logs/RoofSense/profiling")
    # todo check stategies + callbacks + profiler
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            # lightning.pytorch.callbacks.BatchSizeFinder(),
            # lightning.pytorch.callbacks.LearningRateFinder(),
            # lightning.pytorch.callbacks.GradientAccumulationScheduler(scheduling={0: 5}),
            lightning.pytorch.callbacks.LearningRateMonitor(),
        ],
        log_every_n_steps=1,
        logger=logger,
        benchmark=True,
        # profiler=profiler,
        # max_epochs=5
        # detect_anomaly=True
        # fast_dev_run=True
    )
    trainer.fit(model=task, datamodule=datamodule)
