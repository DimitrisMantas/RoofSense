import torch.utils.data
import torchgeo.datasets
import torchgeo.trainers
from torch.utils.data import DataLoader

from roofsense.enums.band import Band
from roofsense.pretraining.contrastive.dataset import ChipDataset


# TODO: Wrap the data loader inside a module to avoid reinitializing the dataset.
def ssl_dataloader() -> DataLoader:
    return DataLoader(
        dataset=ChipDataset(
            dirpath=r"C:\Users\Dimit\Pictures\imgs", bands=Band.ALL[:-1], cache=False
        ),
        batch_size=16,
        shuffle=True,
        num_workers=8,
        collate_fn=torchgeo.datasets.stack_samples,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(0),
        persistent_workers=True,
    )
