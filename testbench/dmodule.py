from typing import Any, Optional, Union

import kornia.augmentation as K
import torch
import torchgeo.datamodules
import torchgeo.samplers
from kornia.constants import DataKey, Resample
from torchgeo.datasets import random_bbox_assignment
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential

from testbench.datasets import RoofSenseDataset


class RoofSenseDataModule(torchgeo.datamodules.GeoDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        patch_size: Union[int, tuple[int, int]] = 224,
        length: Optional[int] = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new GeoDataModule instance.

        Args:
            dataset_class: Class used to instantiate a new dataset.
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__(
            RoofSenseDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )
        # NOTE: Check the Lightning documentation for more information.
        """
        # Data augmentation
        Transform = Callable[[dict[str, Tensor]], dict[str, Tensor]]
        self.aug: Transform = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image"]
        )
        self.train_aug: Optional[Transform] = None
        self.val_aug: Optional[Transform] = None
        self.test_aug: Optional[Transform] = None
        self.predict_aug: Optional[Transform] = None
        """

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = RoofSenseDataset(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = random_bbox_assignment(dataset, [0.6, 0.2, 0.2], generator)

        if stage in ["fit"]:
            self.train_sampler = torchgeo.samplers.PreChippedGeoSampler(
                self.train_dataset,
                # TOSELF: Should the data be reshuffled at the beginning of each epoch?
                shuffle=True,
            )
        if stage in ["fit", "validate"]:
            # TODO: Should this sampler be used in the validation stage?
            self.val_sampler = torchgeo.samplers.GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = torchgeo.samplers.GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )
