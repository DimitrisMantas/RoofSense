import kornia.augmentation as K
import torch.utils.data
from kornia.constants import DataKey, Resample

from roofsense.augmentations.feature import MinMaxScaling


def ssl_augmentations() -> K.AugmentationSequential:
    return K.AugmentationSequential(
        MinMaxScaling(
            mins=torch.tensor([0, 0, 0, 0, 0, -100]),
            maxs=torch.tensor([255, 255, 255, 1, 90, 100]),
        ),
        # TODO: Try the following augmentations:
        # MoCo
        # - K.RandomResizedCrop(size=(size, size), scale=(0.2, 1))
        # - K.RandomBrightness(brightness=(0.6, 1.4), p=0.8)
        # - K.RandomContrast(contrast=(0.6, 1.4), p=0.8
        # - RandomGrayscale
        # - K.RandomGaussianBlur(kernel_size=(ks, ks), sigma=(0.1, 2), p=0.5)
        # SimCLR
        # - K.RandomResizedCrop(size=(size, size), ratio=(0.75, 1.33))
        # - K.RandomBrightness(brightness=(0.2, 1.8), p=0.8)
        # - K.RandomContrast(contrast=(0.2, 1.8), p=0.8)
        # - T.RandomGrayscale(weights=weights, p=0.2)
        # - K.RandomGaussianBlur(kernel_size=(ks, ks), sigma=(0.1, 2))
        # TODO: Figure out if this augmentation is causing representation collapse.
        # K.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.25), p=0.5),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        # TODO: Figure out if this augmentation is causing representation collapse.
        # K.RandomRotation(360),
        # TODO: Try to introduce invariance towards varying lightning conditions.
        data_keys=["image"],
        extra_args={
            DataKey.IMAGE: {"resample": Resample.BILINEAR, "align_corners": False}
        },
    )
