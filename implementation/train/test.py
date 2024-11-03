import warnings
from collections.abc import Iterable

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from roofsense.enums.band import Band
from roofsense.training.datamodule import TrainingDataModule
from roofsense.training.task import TrainingTask


def test_supervised(
    task: TrainingTask,
    datamodule,
    log_dirpath: str,
    study_name: str | None = None,
    experiment_name: int | str | None = None,
    callbacks: Callback | Iterable[Callback] | None = None,
    **kwargs,
) -> Trainer:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.pytorch.seed_everything(42, workers=True)

    logger = TensorBoardLogger(
        save_dir=log_dirpath, name=study_name, version=experiment_name
    )

    cbs = [LearningRateMonitor(), RichProgressBar()]
    if callbacks is not None:
        callbacks = [callbacks] if isinstance(callbacks, Callback) else callbacks
        for cb in callbacks:
            cbs.append(cb)
    trainer = Trainer(logger=logger, callbacks=cbs, benchmark=True, **kwargs)

    with warnings.catch_warnings(action="ignore", category=UserWarning):
        trainer.test(model=task, datamodule=datamodule)

    return trainer


if __name__ == "__main__":
    # pairs = {
    #     "density": (
    #         1,
    #         [
    #             Band.RED,
    #             Band.GREEN,
    #             Band.BLUE,
    #             Band.REFLECTANCE,
    #             Band.SLOPE,
    #             Band.nDRM,  # Band.DENSITY,
    #         ],
    #         [
    #             0,  # red
    #             1,  # green
    #             2,  # blue
    #             3,  # reflectance
    #             4,  # slope
    #             5,  # ndrm
    #             # 6,  # density
    #         ],
    #     ),
    #     "ndrm": (
    #         0,
    #         [
    #             Band.RED,
    #             Band.GREEN,
    #             Band.BLUE,
    #             Band.REFLECTANCE,
    #             Band.SLOPE,
    #             # Band.nDRM,
    #             Band.DENSITY,
    #         ],
    #         [
    #             0,  # red
    #             1,  # green
    #             2,  # blue
    #             3,  # reflectance
    #             4,  # slope
    #             # 5,  # ndrm
    #             6,  # density
    #         ],
    #     ),
    #     "reflectance": (
    #         2,
    #         [
    #             Band.RED,
    #             Band.GREEN,
    #             Band.BLUE,  # Band.REFLECTANCE,
    #             Band.SLOPE,
    #             Band.nDRM,
    #             Band.DENSITY,
    #         ],
    #         [
    #             0,  # red
    #             1,  # green
    #             2,  # blue
    #             # 3,  # reflectance
    #             4,  # slope
    #             5,  # ndrm
    #             6,  # density
    #         ],
    #     ),
    #     "slope": (
    #         0,
    #         [
    #             Band.RED,
    #             Band.GREEN,
    #             Band.BLUE,
    #             Band.REFLECTANCE,  # Band.SLOPE,
    #             Band.nDRM,
    #             Band.DENSITY,
    #         ],
    #         [
    #             0,  # red
    #             1,  # green
    #             2,  # blue
    #             3,  # reflectance
    #             # 4,  # slope
    #             5,  # ndrm
    #             6,  # density
    #         ],
    #     ),
    # }
    pairs = {
        "lidar": (
            0,
            [
                Band.RED,
                Band.GREEN,
                Band.BLUE,
                # Band.REFLECTANCE,
                # Band.SLOPE,
                # Band.nDRM,
                # Band.DENSITY,
            ],
            [
                0,  # red
                1,  # green
                2,  # blue
                # 3,  # reflectance
                # 4,  # slope
                # 5,  # ndrm
                # 6,  # density
            ],
        )
    }
    for name, (version, bands, slice) in pairs.items():
        task = TrainingTask.load_from_checkpoint(
            rf"C:\Documents\RoofSense\logs\ablation_study\{name}_version_{version}\ckpts\last.ckpt",
            model_params={"encoder_params": {"block_args": {"attn_layer": "eca"}}},
        )

        datamodule = TrainingDataModule(
            root="../../dataset/temp", bands=bands, slice=slice
        )

        test_supervised(
            task,
            datamodule,
            log_dirpath="../../logs",
            study_name="ablation_study_test",
            experiment_name=name,
        )
