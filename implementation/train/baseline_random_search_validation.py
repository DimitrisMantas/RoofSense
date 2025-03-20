import numpy as np
import optuna
import torch
from config import Config

from roofsense.enums.band import Band
from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask

if __name__ == "__main__":
    study = optuna.load_study(
        storage="sqlite:///optimization_random_search_round_3_tpe.db",
        study_name="optimization_random_search_round_3",
    )

    best_params = study.best_params
    best_params["learning_rate"] = best_params.pop("lr")

    # best_params = {
    #     k.replace("params_", ""): v
    #     for k, v in best_trial.items()
    #     if k.startswith("params_")
    # }
    # best_params["aa_layer"] = (
    #     None if best_params.get("aa_layer", False) is False else best_params["aa_layer"]
    # )

    config = Config(  # NOTE: This is fixed from the second round.
        annealing="cos",  # NOTE: This is fixed from the first round.
        # base_atrous_rate=1,
        # NOTE: This is fixed from the manual search.
        encoder="resnet18d",  # NOTE: This is fixed from the first round.
        attn_layer="eca",  # NODE: This is fixed from the manual search.
        label_smoothing=0.1,  # NOTE: This is fixed from the second round.
        optimizer="adamw",
        **best_params,
    )

    for i in range(3):
        task = TrainingTask(
            in_channels=6 + 3 * config.append_hsv + config.append_tgi,
            decoder="deeplabv3plus",
            encoder=config.encoder,
            model_params={
                "decoder_atrous_rates": (
                    config.base_atrous_rate,
                    config.base_atrous_rate * 2,
                    config.base_atrous_rate * 3,
                ),
                "encoder_params": {
                    "block_args": {"attn_layer": config.attn_layer},
                    "aa_layer": config.aa_layer,
                    "drop_path_rate": config.drop_path_rate,
                },
            },
            loss_params={
                "names": ["crossentropyloss", "diceloss"],
                "weight": torch.from_numpy(
                    np.fromfile(
                        r"C:\Documents\RoofSense\dataset\temp\weights_tf-idf.bin"
                    )
                ).to(torch.float32),
                "include_background": False,
                "label_smoothing": config.label_smoothing,
            },
            optimizer=config.optimizer,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_epochs=config.warmup_epochs,
            annealing=config.annealing,
        )

        datamodule = TrainingDataModule(
            root="../../dataset/temp",
            bands=[
                Band.RED,
                Band.GREEN,
                Band.BLUE,
                # Band.REFLECTANCE,
                Band.SLOPE,
                Band.nDRM,
                Band.DENSITY,
            ],
            append_hsv=config.append_hsv,
            append_tgi=config.append_tgi,
        )

        train_supervised(
            task,
            datamodule,
            log_dirpath="../../logs",
            study_name="ablation_study",
            experiment_name=f"reflectance_version_{i}",
            max_epochs=200,
            test=False,
        )
