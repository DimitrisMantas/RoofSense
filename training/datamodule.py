import copy
import os
import warnings

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from typing_extensions import override

from common.augmentations import AppendGreenness, MinMaxScaling
from training.dataset import TrainingDataset


class TrainingDataModule(NonGeoDataModule):
    # TODO: Move these to TrainingDataset so that they can be constructed according
    #  to the specified band list. The minimum cell value of each raster stack layer.
    mins = torch.tensor([0, 0, 0, 0, 0])

    # The maximum cell value of each raster stack layer.
    maxs = torch.tensor([255, 255, 255, 1, 90])

    def __init__(
        self,
        batch_size: int = 8,
        lengths=(0.7, 0.15, 0.15),
        num_workers: int | None = None,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            TrainingDataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.lengths = lengths
        max_workers = max(os.cpu_count() // 2, 1)
        if num_workers is None:
            self.num_workers = max_workers
        elif num_workers > num_workers:
            msg = (
                "The total number of  data loader worker threads may be too large. "
                "This can result in in abnormally large amount of virtual memory "
                "being registered, which can in-turn lead to erroneous behavior and "
                "out-of-memory errors."
                + "\n"
                + "Monitor memory consumption during the model fitting stage and "
                "consider decreasing the total number of  data loader worker "
                "threads to at most half the count of logical processors on your "
                "machine."
                + "\n"
                + "See https://tinyurl.com/439cb683 and https://tinyurl.com/yc63wxey "
                "for  more information."
            )
            warnings.warn(msg, UserWarning)

        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        # General Augmentations
        self.aug = AugmentationSequential(
            # Scaling
            MinMaxScaling(self.mins, self.maxs),
            # Spectral Indices
            AppendGreenness(),
            data_keys=["image", "mask"],
            extra_args={
                # TODO: Figure out what the most appropriate setting for
                #  `align_corners` should be.
                DataKey.IMAGE: {"resample": Resample.BILINEAR, "align_corners": None},
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None},
            },
        )

        # Training Augmentations
        # NOTE: This field overwrites the predefined augmentations.
        self.train_aug = AugmentationSequential(
            # Scaling
            MinMaxScaling(self.mins, self.maxs),
            # Spectral Indices
            AppendGreenness(),
            # Geometric Augmentations
            # NOTE: These augmentations correspond to the D4 dihedral group.
            # Flips
            K.RandomVerticalFlip(), K.RandomHorizontalFlip(),
            # RandomDiagonalFlip(diag="main"),
            # RandomDiagonalFlip(diag="anti"),
            # Rotations
            K.RandomRotation((90, 90)),
            K.RandomRotation((90, 90)),
            K.RandomRotation((90, 90)),
            # Intensity Augmentations
            # RandomSharpness(0.1),
            # ColorJiggle(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            data_keys=["image", "mask"],
            extra_args={
                # TODO: Figure out what the most appropriate setting for
                #  `align_corners` should be.
                DataKey.IMAGE: {"resample": Resample.BILINEAR, "align_corners": None},
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None},
            },
        )

    @override
    def setup(self, stage: str) -> None:
        dataset = TrainingDataset(**self.kwargs)

        subsets = random_split(
            dataset, lengths=self.lengths, generator=torch.Generator().manual_seed(0)
        )

        datasets = []
        for subset in subsets:
            temp = copy.deepcopy(dataset)
            temp.img_paths = [subset.dataset.img_paths[i] for i in subset.indices]
            temp.msk_paths = [subset.dataset.msk_paths[i] for i in subset.indices]

            datasets.append(temp)

        self.train_dataset, self.val_dataset, self.test_dataset = datasets

    @override
    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        return DataLoader(
            dataset=getattr(self, f"{split}_dataset"),
            batch_size=self.batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(0),
            persistent_workers=self.persistent_workers,
        )
