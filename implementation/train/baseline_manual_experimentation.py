import numpy as np
import torch

from data import Config
from roofsense.runners import train_supervised
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask

trial_names = [
    # "sanity_check",
    # "input_data_append_hsv",
    # "input_data_append_tgi",
    # "lr_scheduler_annealing_cos",
    # "lr_scheduler_warmup_epochs_50",
    # "lr_scheduler_warmup_epochs_100",
    # "lr_scheduler_warmup_epochs_150",
    "model_decoder_base_atrous_rate_1",
    "model_decoder_base_atrous_rate_10",
    "model_decoder_base_atrous_rate_15",
    "model_decoder_base_atrous_rate_20",
    # "model_encoder_anti_aliasing",
    # "model_encoder_attention_eca",
    # "model_encoder_attention_se",
    # "model_encoder_resnet18d",
    # "optimizer_adamw",
    # "optimizer_learning_rate_1e-2",
    # "optimizer_learning_rate_1e-4",
    # "optimizer_learning_rate_5e-3",
    # "optimizer_learning_rate_5e-4",
    # "optimizer_weight_decay_1e-1",
    # "optimizer_weight_decay_1e-2",
    # "optimizer_weight_decay_1e-3",
    "regularization_label_smoothing_005",
    "regularization_label_smoothing_010",
    "regularization_label_smoothing_015",
    # "regularization_stochastic_depth_005",
    # "regularization_stochastic_depth_010",
    # "regularization_stochastic_depth_015",
]

trial_configs = [
    # Config(),
    # Config(append_hsv=True),
    # Config(append_tgi=True),
    # Config(annealing="cos"),
    # Config(warmup_epochs=50),
    # Config(warmup_epochs=100),
    # Config(warmup_epochs=150),
    Config(base_atrous_rate=1),
    Config(base_atrous_rate=10),
    Config(base_atrous_rate=15),
    Config(base_atrous_rate=20),
    # Config(aa_layer=timm.layers.BlurPool2d),
    # Config(attn_layer="eca"),
    # Config(attn_layer="se"),
    # Config(encoder="resnet18d"),
    # Config(optimizer="adamw"),
    # Config(learning_rate=1e-2),
    # Config(learning_rate=1e-4),
    # Config(learning_rate=5e-3),
    # Config(learning_rate=5e-4),
    # Config(weight_decay=1e-1),
    # Config(weight_decay=1e-2),
    # Config(weight_decay=1e-3),
    Config(label_smoothing=0.05),
    Config(label_smoothing=0.10),
    Config(label_smoothing=0.15),
    # Config(drop_path_rate=0.05),
    # Config(drop_path_rate=0.10),
    # Config(drop_path_rate=0.15),
]

trials: dict[str, Config] = dict(zip(trial_names, trial_configs, strict=True))

if __name__ == "__main__":
    for name, config in trials.items():
        task = TrainingTask(
            in_channels=7 + 3 * config.append_hsv + config.append_tgi,
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
            append_hsv=config.append_hsv,
            append_tgi=config.append_tgi,
        )

        train_supervised(
            task,
            datamodule,
            log_dirpath="../../logs",
            study_name="hyperparameter_optimization_manual_experimentation",
            experiment_name=f"{name}_version_1",
            max_epochs=200,
            test=False,
        )
