from __future__ import annotations

from typing import Optional

import kornia.augmentation as K
import torch
from torch import Tensor


class AppendEXG(K.IntensityAugmentationBase2D):
    """Basically NDVI for RGB (https://research.tudelft.nl/files/96224757/1_s2.0_S0924271621001854_main.pdf). rgb data must be 0..1"""

    def __init__(
        self, index_red: int = 0, index_green: int = 1, index_blue: int = 2
    ) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: reference band channel index
            index_green: difference band channel index of component 1
            index_blue: difference band channel index of component 2
        """
        super().__init__(p=1)
        self.flags = {
            "index_red": index_red,
            "index_green": index_green,
            "index_blue": index_blue,
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_red = input[..., flags["index_red"], :, :]
        band_green = input[..., flags["index_green"], :, :]
        band_blue = input[..., flags["index_blue"], :, :]

        ag = 2 * band_green - band_blue - band_red
        # NOTE: This is [-1,1] (the range is known because the index is computed after the scaling step) so we need to scale it to [0,1]
        ag = (ag + 2) / 4

        # add the channel dim to be able to cat
        ag = ag.unsqueeze(dim=1)

        input = torch.cat((input, ag), dim=1)
        return input


class AppendNDRGI(K.IntensityAugmentationBase2D):
    """Basically NDVI for RGB (https://research.tudelft.nl/files/96224757/1_s2.0_S0924271621001854_main.pdf). rgb data must be 0..1"""

    def __init__(
        self, index_red: int = 0, index_green: int = 1, index_blue: int = 2
    ) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: reference band channel index
            index_green: difference band channel index of component 1
            index_blue: difference band channel index of component 2
        """
        super().__init__(p=1)
        self.flags = {
            "index_red": index_red,
            "index_green": index_green,
            "index_blue": index_blue,
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_red = input[..., flags["index_red"], :, :]
        band_green = input[..., flags["index_green"], :, :]
        band_blue = input[..., flags["index_blue"], :, :]

        ag = (band_red - band_green) / (band_red + band_green)
        # NOTE: This is [-1,1] (the range is known because the index is computed after the scaling step) so we need to scale it to [0,1]
        ag = (ag + 1) / 2

        # add the channel dim to be able to cat
        ag = ag.unsqueeze(dim=1)

        input = torch.cat((input, ag), dim=1)
        return input


class AppendTGI(K.IntensityAugmentationBase2D):
    """Basically NDVI for RGB (https://research.tudelft.nl/files/96224757/1_s2.0_S0924271621001854_main.pdf). rgb data must be 0..1"""

    def __init__(
        self, index_red: int = 0, index_green: int = 1, index_blue: int = 2
    ) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: reference band channel index
            index_green: difference band channel index of component 1
            index_blue: difference band channel index of component 2
        """
        super().__init__(p=1)
        self.flags = {
            "index_red": index_red,
            "index_green": index_green,
            "index_blue": index_blue,
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_red = input[..., flags["index_red"], :, :]
        band_green = input[..., flags["index_green"], :, :]
        band_blue = input[..., flags["index_blue"], :, :]

        ag = band_green - 0.39 * band_red - 0.61 * band_blue
        # NOTE: This is [-1,1] (the range is known because the index is computed after the scaling step) so we need to scale it to [0,1]
        ag = (ag + 1) / 2

        # add the channel dim to be able to cat
        ag = ag.unsqueeze(dim=1)

        input = torch.cat((input, ag), dim=1)
        return input


class AppendVIDVI(K.IntensityAugmentationBase2D):
    """Basically NDVI for RGB (https://research.tudelft.nl/files/96224757/1_s2.0_S0924271621001854_main.pdf). rgb data must be 0..1"""

    def __init__(
        self, index_red: int = 0, index_green: int = 1, index_blue: int = 2
    ) -> None:
        """Initialize a new transform instance.

        Args:
            index_red: reference band channel index
            index_green: difference band channel index of component 1
            index_blue: difference band channel index of component 2
        """
        super().__init__(p=1)
        self.flags = {
            "index_red": index_red,
            "index_green": index_green,
            "index_blue": index_blue,
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        band_red = input[..., flags["index_red"], :, :]
        band_green = input[..., flags["index_green"], :, :]
        band_blue = input[..., flags["index_blue"], :, :]

        ag = (2 * band_green - band_blue - band_red) / (
            2 * band_green + band_blue + band_red
        )
        # NOTE: This is [-1,1] (the range is known because the index is computed after the scaling step) so we need to scale it to [0,1]
        ag = (ag + 1) / 2

        # add the channel dim to be able to cat
        ag = ag.unsqueeze(dim=1)

        input = torch.cat((input, ag), dim=1)
        return input
