import warnings
from collections import OrderedDict
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Conv2d


def get_encoder_params(
    ckpt_path: str,
    param_name: str | Iterable[str] = ("backbone", "encoder"),
    state_dict_prefix: str = "model.encoder.",
) -> tuple[str, OrderedDict[str, Tensor]]:
    """Extract the name and state dictionary of an encoder from a PyTorch Lightning
    checkpoint.

    Args:
        ckpt_path:
            The path to the checkpoint.
        param_name:
            The name of the model parameter corresponding to the encoder. If more
            than one names are provided, each one will be compared against the
            corresponding checkpoint data to determine the actual parameter name. If
            this search does not return any matches, a relevant warning is issued.
        state_dict_prefix:
            The data prepended to each key of the state dictionary by a larger model
            based on the encoder.

    Returns:
        The name and state dictionary of the encoder. If the name could not be
        resolved, ``None`` is returned in its place, instead.
    """
    ckpt = torch.load(ckpt_path)

    # Try to
    if isinstance(param_name, str):
        param_name = [param_name]
    name: str | None = None
    for param in param_name:
        if param in ckpt["hyper_parameters"]:
            name = ckpt["hyper_parameters"][param]
            break
    if name is None:
        msg = (
            f"Found no hyperparameter with name in {param_name!r} in specified "
            f"checkpoint. Cannot resolve encoder name."
        )
        warnings.warn(msg, UserWarning)

    weights: OrderedDict[str, Tensor] = ckpt["state_dict"]
    weights = OrderedDict(
        {name: value for name, value in weights.items() if state_dict_prefix in name}
    )
    weights = OrderedDict(
        {name.replace(state_dict_prefix, ""): value for name, value in weights.items()}
    )

    return name, weights


def reinit_initial_conv_layer(
    layer: Conv2d,
    new_in_channels: int,
    keep_first_n_weights: int | None,
    new_stride: int | tuple[int, int] | None = None,
    new_padding: str | int | tuple[int, int] | None = None,
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
        kernel_size=layer.kernel_size,
        # type: ignore[arg-type]
        stride=updated_stride,
        # type: ignore[arg-type]
        padding=updated_padding,
        # type: ignore[arg-type]
        dilation=layer.dilation,
        # type: ignore[arg-type]
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
