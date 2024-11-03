import numpy as np
import torch

from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask

if __name__ == "__main__":
    for _ in range(3):
        task = TrainingTask(
            decoder="deeplabv3plus",
            encoder="resnet18",
            model_params={"decoder_atrous_rates": (6, 12, 18)},
            loss_params={
                "names": ["crossentropyloss", "diceloss"],
                "weight": torch.from_numpy(
                    np.fromfile(
                        r"C:\Documents\RoofSense\dataset\temp\weights_tf-idf.bin"
                    )
                ).to(torch.float32),
                "include_background": False,
            },
        )

        datamodule = TrainingDataModule(root="../../dataset/temp")

        train_supervised(
            task,
            datamodule,
            log_dirpath="../../logs",
            study_name="baseline",
            max_epochs=300,
            test=False,
        )
