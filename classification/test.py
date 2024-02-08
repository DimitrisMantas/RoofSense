from lightning import Trainer
from torchgeo.trainers import SemanticSegmentationTask

from classification.datamodules import TrainingDataModule

datamodule = TrainingDataModule(  # Dataset Options
    paths="../pretraining",  # Data Module Options
    batch_size=8,
    patch_size=128,
    length=1024,
    num_workers=10,
)
task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet18",
    weights=True,
    in_channels=6,
    num_classes=9,
    loss="ce",
    ignore_index=None,
    lr=0.1,
    patience=6,
)
# Training must be performed under a guard.
# https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
if __name__ == "__main__":
    trainer = Trainer(default_root_dir="logs/RoofSense", max_epochs=10)

    trainer.fit(model=task, datamodule=datamodule)
