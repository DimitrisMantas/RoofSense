# %%

import kornia.augmentation as K
import lightning
import torch.utils.data
import torchgeo.datasets
import torchgeo.trainers
from kornia.constants import DataKey, Resample

from augmentations.feature import MinMaxScaling
from enums.band import Band
from pretraining.unsupervised.dataset import ChipDataset


# %%
def ssl_augmentations() -> K.AugmentationSequential:
    return K.AugmentationSequential(
        MinMaxScaling(
            mins=torch.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    -100,  # 0
                ]
            ),
            maxs=torch.tensor(
                [
                    255,
                    255,
                    255,
                    1,
                    90,
                    100,  # 100
                ]
            ),
        ),
        K.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.25), p=0.5),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        K.RandomRotation((90, 90)),
        K.RandomRotation((90, 90)),
        K.RandomRotation((90, 90)),
        # TODO: Try to introduce invariance towards varying lightning conditions.
        data_keys=["image"],
        extra_args={
            DataKey.IMAGE: {"resample": Resample.BILINEAR, "align_corners": False}
        },
    )


# %%
task = torchgeo.trainers.MoCoTask(
    # TODO: Load the encoder using torchseg or at least make sure that it is supported downstream.
    model="resnet18",
    # TODO: Drop support for custom weights.
    weights=True,
    in_channels=6,
    lr=0.6 * 8 / 256,
    augmentation1=ssl_augmentations(),
    augmentation2=ssl_augmentations(),
)
# %%
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.seed_everything(0, workers=True)

    # TODO: Wrap the data loader inside a module to avoid reinitializing the dataset.
    dataloader = torch.utils.data.DataLoader(
        dataset=ChipDataset(
            dirpath=r"C:\Users\Dimit\Pictures\imgs", bands=Band.ALL[:-1], cache=False
        ),
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=torchgeo.datasets.stack_samples,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(0),
        persistent_workers=False,
    )

    trainer = lightning.Trainer(max_epochs=200, benchmark=True)
    trainer.fit(task, train_dataloaders=dataloader)

    torch.save(task.backbone.state_dict(), "resnet18_mocov3_v0")
