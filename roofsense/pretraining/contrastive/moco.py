import lightning
import torch.utils.data
import torchgeo.trainers

from roofsense.pretraining.contrastive.augmentations import ssl_augmentations
from roofsense.pretraining.contrastive.dataloader import ssl_dataloader

# v2: only flips
# v3: flips and my crops


task = torchgeo.trainers.MoCoTask(
    # TODO: Load the encoder using torchseg or at least make sure that it is
    #  supported downstream.
    model="resnet18",
    # TODO: Drop support for custom weights.
    weights=True,
    in_channels=6,
    lr=0.6 * 8 / 256,
    augmentation1=ssl_augmentations(),
    augmentation2=ssl_augmentations(),
)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    lightning.seed_everything(0, workers=True)

    trainer = lightning.Trainer(max_epochs=200, benchmark=True)
    trainer.fit(task, train_dataloaders=ssl_dataloader())

    torch.save(task.backbone.state_dict(), "resnet18_mocov3_v2")
