import os
import warnings

import kornia.augmentation as K
import lightning
import torch
from kornia.constants import DataKey, Resample
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.transforms import AugmentationSequential
from torchgeo.transforms.transforms import _RandomNCrop

from roofsense.augmentations.feature import MinMaxScaling
from roofsense.training.task import TrainingTask

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(0, workers=True)

    datamodule = InriaAerialImageLabelingDataModule(
        root=r"C:\Users\Dimit\Downloads\aerialimagelabeling",
        batch_size=2,
        patch_size=512,
        num_workers=2,
    )
    datamodule.aug = AugmentationSequential(
        MinMaxScaling(mins=torch.tensor([0] * 3), maxs=torch.tensor([255] * 3)),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        _RandomNCrop(
            datamodule.patch_size,
            # TODO: Store the actual batch size for later use.
            2,
        ),
        data_keys=["image", "mask"],
        extra_args={
            DataKey.IMAGE: {"resample": Resample.BILINEAR, "align_corners": False},
            DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": False},
        },
    )
    datamodule.train_aug = datamodule.predict_aug = None

    task = TrainingTask(
        decoder="deeplabv3plus",
        encoder="resnet18",
        loss_params={
            "names": [
                "crossentropyloss"
                # , "diceloss"
            ]
            # "weight": torch.from_numpy(np.fromfile("../dataset/temp/weights.bin")).to(
            #     torch.float32
            # ),
            # "label_smoothing": 0.05,
            # "squared_pred": True,
        },
        in_channels=3,
        num_classes=2,
        warmup_epochs=5,
        T_0=95,
        init_lr_pct=1e-6,
    )

    logger = TensorBoardLogger(
        save_dir="../../../../logs",
        name="pretraining",
        version="inria-aerial-image-labelling_batch-size-2_max-epochs-100",
    )
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "ckpts"),
        filename="best",
        monitor="val/loss",
        save_last=True,
    )
    # Match log and checkpoint version numbers in the case of automatic versioning.
    model_ckpt.STARTING_VERSION = 0

    trainer = Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), model_ckpt, RichProgressBar()],
        max_epochs=100,
        benchmark=True,
    )

    with warnings.catch_warnings(action="ignore", category=UserWarning):
        trainer.fit(model=task, datamodule=datamodule)
