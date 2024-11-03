import os
import warnings

import kornia.augmentation as K
import numpy as np
import torch
from kornia.constants import DataKey, Resample
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from typing_extensions import override

from roofsense.augmentations.color import AppendHSV
from roofsense.augmentations.feature import MinMaxScaling
from roofsense.augmentations.index import AppendTGI
from roofsense.training.dataset import TrainingDataset


class TrainingDataModule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        append_hsv: bool = False,
        append_tgi: bool = False,
        num_workers: int | None = None,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        slice=None,
        **kwargs,
    ) -> None:
        super().__init__(
            TrainingDataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        max_num_workers = max(1, os.cpu_count() // 2)
        if num_workers is None:
            self.num_workers = min(batch_size, max_num_workers)
        elif batch_size < num_workers <= max_num_workers:
            msg = (
                f"Number of specified worker processes: {num_workers!r} greater than "
                f"batch size: {batch_size!r}. Improved performance due to excess "
                f"workers is not guaranteed."
            )
            warnings.warn(msg, UserWarning)
        else:
            msg = (
                f"Number of specified worker processes: {num_workers!r} greater than "
                f"half the total number of logical CPUs in the system: "
                f"{max_num_workers!r}. Windows systems may suffer from erroneous "
                f"behaviour and out-of-memory errors due to excessive memory paging. "
                f"See See https://tinyurl.com/439cb683 and "
                f"https://tinyurl.com/yc63wxey for more information."
            )
            warnings.warn(msg, ResourceWarning)

        if self.num_workers == 0 and persistent_workers:
            msg = (
                "No worker processes specified to be persisted. Value of "
                "'persistent_workers' will be set to 'False'."
            )
            warnings.warn(msg, UserWarning)
            self.persistent_workers = False
        else:
            self.persistent_workers = persistent_workers

        self.pin_memory = pin_memory

        mins, maxs = torch.tensor_split(
            torch.from_numpy(
                np.fromfile(
                    os.path.join(kwargs["root"], TrainingDataset.scales_filename)
                )
            ),
            2,
        )

        self.slice = (
            [
                0,  # red
                1,  # green
                2,  # blue
                3,  # reflectance
                4,  # slope
                5,  # ndrm
                6,  # density
            ]
            if slice is None
            else slice
        )
        args = [
            # Scaling
            MinMaxScaling(
                # TODO: Perform band filtering automatically.
                mins[self.slice],
                maxs[self.slice],
            )
        ]
        # Color spaces & Spectral Indices
        if append_hsv:
            args.append(AppendHSV())
        if append_tgi:
            args.append(AppendTGI())

        kwargs = {
            "data_keys": ["image", "mask"],
            "extra_args": {
                # NOTE: We choose to always resample with bilinear interpolation to
                # preserve the scaled value range of the stack.
                # This is important because both reflectance and slope values have a
                # physical interpretation.
                # NOTE; Interpolation with aligned corners may disturb the spatial
                # inductive biases of the model.
                # See https://discuss.pytorch.org/t/what-we-should-use-align-corners
                # -false/22663/5 for more information.
                DataKey.IMAGE: {"resample": Resample.BILINEAR, "align_corners": False},
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": False},
            },
        }

        # Validation & Test/Prediction Augmentations
        self.aug = AugmentationSequential(*args, **kwargs)

        args += [
            # Geometric Augmentations
            # Flips
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            # Rotations
            K.RandomRotation((90, 90)),
            K.RandomRotation((90, 90)),
            K.RandomRotation((90, 90)),
        ]

        # Training Augmentations
        self.train_aug = AugmentationSequential(*args, **kwargs)

    @override
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(split="training", **self.kwargs)
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(split="validation", **self.kwargs)
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(split="test", **self.kwargs)

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
