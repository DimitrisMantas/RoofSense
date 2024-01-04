import lightning.pytorch
import torchgeo.datamodules
import torchgeo.datasets
import torchgeo.samplers
import torchgeo.trainers
import torchgeo.transforms

import testbench


def main():
    # NOTE: Lowering the multiplication precision allows for more efficient use of the
    #       Tensor Cores found in certain CUDA devices at the cost of relatively
    #       degraded numerical precision.
    # torch.set_float32_matmul_precision("high")

    datamodule = testbench.model.datamodules.L8BiomeDataModule(
        paths="data/l7irish",
        # NOTE: The batch size is device-specific and must be specified empirically such
        #       that the available memory space not exhausted.
        batch_size=16,
        num_workers=8,
        download=True,
    )

    task = torchgeo.trainers.SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        weights=True,
        in_channels=3,
        # NOTE: The total number of prediction classes is equal to the number of
        #       individual classes plus the background.
        num_classes=5,
        loss="ce",
        ignore_index=None,
        lr=0.1,
        patience=6,
    )

    trainer = lightning.pytorch.Trainer()
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()
