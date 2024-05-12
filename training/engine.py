import lightning.pytorch
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (EarlyStopping,
                                         LearningRateMonitor,
                                         ModelCheckpoint, )
from lightning.pytorch.loggers import TensorBoardLogger

from training.datamodule import TrainingDataModule
from training.task import TrainingTask

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(42, workers=True)

    task = TrainingTask(  # Model Configuration
        model="unet",
        backbone="resnet18",
        # ImageNet
        weights=True,
        in_channels=5,
        num_classes=8 + 1,
        # model_kwargs={
        #     "decoder_attention_type": "scse"
        # },
        # Loss Configuration
        loss="CrossEntropyJaccard",
        class_weights=torch.tensor(
            np.load("../dataset/temp/weights.npy"), dtype=torch.float32
        ),
        ignore_index=0,
    )

    datamodule = TrainingDataModule(  # TODO: Try a batch size of 12.
        root="../dataset/temp", batch_size=16, num_workers=8
    )

    # todo check strategies + callbacks + profiler
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir="../logs/RoofSense"),
        callbacks=[
            ModelCheckpoint(
                dirpath="../logs/RoofSense",
                filename="best",
                monitor="val/loss",
                save_last=True,
            ),
            EarlyStopping(monitor="val/loss", patience=1000),
            LearningRateMonitor(),
        ],
        log_every_n_steps=1,
        benchmark=True
    )

    trainer.fit(model=task, datamodule=datamodule)
