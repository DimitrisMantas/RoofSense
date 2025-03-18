import numpy as np
import torch

from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask

if __name__ == "__main__":
    # for _ in range(3):
    task = TrainingTask(
        encoder="tu-resnet18",
        decoder="deeplabv3plus",
        model_cfg={"decoder_atrous_rates": (6, 12, 18), "decoder_aspp_dropout": 0},
        loss_cfg={
            "names": ["crossentropyloss", "diceloss"],
            "weight": torch.from_numpy(
                np.fromfile(r"C:\Documents\RoofSense\dataset\temp\weights_tf-idf.bin")
            ).to(torch.float32),
            "include_background": False,
        },
        optimizer_cfg={"eps": 1e-7},
        scheduler_cfg={"total_iters": 300, "power": 0.9},
    )

    datamodule = TrainingDataModule(root="../../dataset/temp", auto_drop_last=True)

    train_supervised(
        task,
        datamodule,
        log_dirpath="../../logs",
        study_name="perftest",
        experiment_name="resnet18_newtask",
        max_epochs=300,
        test=False,
    )
