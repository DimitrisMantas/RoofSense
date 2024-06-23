from typing import Optional, Union, cast

import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Conv2d


def reinit_initial_conv_layer(
    layer: Conv2d,
    new_in_channels: int,
    keep_first_n_weights: int|None,
    new_stride: Optional[Union[int, tuple[int, int]]] = None,
    new_padding: Optional[Union[str, Union[int, tuple[int, int]]]] = None,
) -> Conv2d:
    """Clones a Conv2d layer while optionally retaining some of the original weights.

    When replacing the first convolutional layer in a model with one that operates over
    different number of input channels, we sometimes want to keep a subset of the kernel
    weights the same (e.g. the RGB weights of an ImageNet pretrained model). This is a
    convenience function that performs that function.

    Args:
        layer: the Conv2d layer to initialize
        new_in_channels: the new number of input channels
        keep_rgb_weights: flag indicating whether to re-initialize the first 3 channels
        new_stride: optionally, overwrites the ``layer``'s stride with this value
        new_padding: optionally, overwrites the ``layers``'s padding with this value

    Returns:
        a Conv2d layer with new kernel weights
    """
    use_bias = layer.bias is not None
    if keep_first_n_weights is not None:
        w_old = layer.weight.data[:, :keep_first_n_weights, :, :].clone()
        if use_bias:
            b_old = cast(Tensor, layer.bias).data.clone()

    updated_stride = layer.stride if new_stride is None else new_stride
    updated_padding = layer.padding if new_padding is None else new_padding

    new_layer = Conv2d(
        new_in_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,  # type: ignore[arg-type]
        stride=updated_stride,  # type: ignore[arg-type]
        padding=updated_padding,  # type: ignore[arg-type]
        dilation=layer.dilation,  # type: ignore[arg-type]
        groups=layer.groups,
        bias=use_bias,
        padding_mode=layer.padding_mode,
    )
    nn.init.kaiming_normal_(new_layer.weight, mode="fan_out", nonlinearity="relu")

    if keep_first_n_weights is not None:
        new_layer.weight.data[:, :keep_first_n_weights, :, :] = w_old
        if use_bias:
            cast(Tensor, new_layer.bias).data = b_old

    return new_layer