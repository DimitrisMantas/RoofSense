from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import kornia.augmentation as K
import torch
from overrides import override
from torch import Tensor


class MinMaxScaling(K.IntensityAugmentationBase2D):
    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1, same_on_batch=True)
        self.flags = {"mins": mins.view(1, -1, 1, 1), "maxs": maxs.view(1, -1, 1, 1)}
        self.delta = 1e-10

    # noinspection PyShadowingBuiltins
    @override
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        mins = torch.as_tensor(flags["mins"], device=input.device, dtype=input.dtype)
        maxs = torch.as_tensor(flags["maxs"], device=input.device, dtype=input.dtype)
        return (input - mins) / (maxs - mins + self.delta)

    # noinspection PyShadowingBuiltins
    def apply_transform_mask(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        return input


# TODO: Check whether it is possible to have an augmentation factory which will
#       perform the necessary band filtering, avoiding duplicated code fragments.
class RandomBrightness(K.RandomBrightness):
    def __init__(self, brightness: Tuple[float, float] = (1.0, 1.0), p: float = 1.0):
        super().__init__(brightness, p=p)

    # noinspection PyShadowingBuiltins
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Transform only the RGB bands of the input.
        # TODO: Consider getting the appropriate band indices automatically from the
        #       corresponding dataset specification.
        color: Tensor = input[:, [0, 1, 2], ...]
        color = super().apply_transform(color, params, flags, transform)

        # FIXME: Check whether it is possible to avoid data copies.
        input[:, [0, 1, 2], ...] = color

        return input


class RandomGamma(K.RandomGamma):
    def __init__(
        self,
        gamma: Tuple[float, float] = (1.0, 1.0),
        gain: Tuple[float, float] = (1.0, 1.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(gamma, gain, p=p)

    # noinspection PyShadowingBuiltins
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Transform only the RGB bands of the input.
        # TODO: Consider getting the appropriate band indices automatically from the
        #       corresponding dataset specification.
        color: Tensor = input[:, [0, 1, 2], ...]
        color = super().apply_transform(color, params, flags, transform)

        # FIXME: Check whether it is possible to avoid data copies.
        input[:, [0, 1, 2], ...] = color

        return input


class RandomSaturation(K.RandomSaturation):
    def __init__(self, saturation: Tuple[float, float] = (1.0, 1.0), p: float = 1.0):
        super().__init__(saturation, p=p)

    # noinspection PyShadowingBuiltins
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Transform only the RGB bands of the input.
        # TODO: Consider getting the appropriate band indices automatically from the
        #       corresponding dataset specification.
        color: Tensor = input[:, [0, 1, 2], ...]
        color = super().apply_transform(color, params, flags, transform)

        # FIXME: Check whether it is possible to avoid data copies.
        input[:, [0, 1, 2], ...] = color

        return input


class RandomHue(K.RandomHue):
    def __init__(self, hue: Tuple[float, float] = (1.0, 1.0), p: float = 1.0):
        super().__init__(hue, p=p)

    # noinspection PyShadowingBuiltins
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Transform only the RGB bands of the input.
        # TODO: Consider getting the appropriate band indices automatically from the
        #       corresponding dataset specification.
        color: Tensor = input[:, [0, 1, 2], ...]
        color = super().apply_transform(color, params, flags, transform)

        # FIXME: Check whether it is possible to avoid data copies.
        input[:, [0, 1, 2], ...] = color

        return input
