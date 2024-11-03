from roofsense.pretraining.supervised.potsdam.dataset import PotsdamRGBDataset
from roofsense.pretraining.supervised.potsdam.dtmodul import PotsdamRGBDataModule
from roofsense.runners import train_supervised
from roofsense.training.task import TrainingTask

if __name__ == "__main__":
    task = TrainingTask(
        in_channels=3,
        num_classes=6,
        decoder="deeplabv3plus",
        encoder="resnet18d",
        model_params={
            "decoder_atrous_rates": (6, 12, 18),
            "encoder_params": {"block_args": {"attn_layer": "eca"}},
        },
        loss_params={"names": ["crossentropyloss", "diceloss"]},
    )

    datamodule = PotsdamRGBDataModule(
        dataset_class=PotsdamRGBDataset,
        root=r"C:\Documents\RoofSense\dataset\pretraining",
    )

    train_supervised(
        task,
        datamodule,
        log_dirpath="../../logs",
        study_name="pretraining",
        experiment_name="max_epochs_5",
        max_epochs=5,
        test=False,
    )
