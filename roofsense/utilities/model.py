import warnings
from collections import OrderedDict
from functools import reduce
from typing import List, Literal, Never, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def load_model_from_lightning_checkpoint(filename: str) -> OrderedDict[str, Tensor]:
    checkpoint = torch.load(filename, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    state_dict = OrderedDict({k: v for k, v in state_dict.items() if "model." in k})
    state_dict = OrderedDict(
        {k.replace("model.", ""): v for k, v in state_dict.items()}
    )
    return state_dict


def get_conv_output_shapes(
    model: nn.Module, input_shape: tuple[int, int, int], preceding_norm: bool = False
) -> list[tuple[int, ...]]:
    """Compute the output shape of each convolution layer of a given model with respect to a particular input tensor shape.

    Args:
        model:
            The model.
        input_shape:
            The input shape (C, H, W).
            The batch dimension is handled automatically.
        preceding_norm:
            True to compute the output shape of the layers preceding a batch normalization layer; False otherwise.

    Returns:
        The output shape of each layer in the order it is defined in the model.
        The batch dimension is excluded.
    """
    is_training = model.training
    if model.training:
        model.eval()

    dummy_input = torch.zeros(
        (  # _BatchNorm expects multiple samples.
            1,
            *input_shape,
        )
    )
    shapes: list[tuple[int, ...]] = []

    def hook(
        module: nn.Module | Never,
        input: Tuple[torch.Tensor] | Never,
        output: torch.Tensor,
    ) -> None:
        shapes.append(output.shape[1:])

    hooks = []
    modules = list(model.modules())
    for i, module in enumerate(modules):
        if isinstance(module, (nn.modules.conv._ConvNd)) and (
            preceding_norm
            and i + 1 < len(modules)
            and isinstance(modules[i + 1], nn.modules.batchnorm._BatchNorm)
        ):
            hooks.append(module.register_forward_hook(hook))

    with torch.inference_mode():
        model(dummy_input)

    for h in hooks:
        h.remove()

    if is_training:
        model.train()

    return shapes


def set_module_by_given_name(model: nn.Module, name: str, module: nn.Module) -> None:
    """Replace a layer of a given model with a particular name.

    Args:
        model:
            The model.
        name:
            The name of the layer to replace.
        module:
            The new layer.
    """
    components = name.split(".")
    parent_block = reduce(getattr, components[:-1], model)
    setattr(parent_block, components[-1], module)


def replace_bn(model: nn.Module, norm_cls: type[nn.Module], **kwargs) -> None:
    """Replace the batch normalization layers of a given model with equivalent layers of a different type.

    Args:
        model:
            The model.
        norm_cls:
            The new layer.
            Only torch.nn.BatchNorm2d, torch.nn.GroupNorm, and torch.nnLayerNorm are currently supported.
        **kwargs:
            The initializer arguments of the new layer.
    """
    old_named_modules = dict(model.named_modules())

    modules: OrderedDict[str, nn.modules.batchnorm._BatchNorm] = OrderedDict(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    if issubclass(norm_cls, nn.LayerNorm):
        input_shape = kwargs.pop("input_shape", None)
        if input_shape is None:
            raise ValueError(
                f"The shape of the input tensor to the model must be provided in order to initialize {norm_cls!r}."
            )
        conv_output_shapes = get_conv_output_shapes(
            model, input_shape, preceding_norm=True
        )
        for name, shape in zip(modules.keys(), conv_output_shapes, strict=True):
            set_module_by_given_name(model, name, norm_cls(shape, **kwargs))
    elif issubclass(norm_cls, nn.GroupNorm):
        for name, module in modules.items():
            set_module_by_given_name(
                model, name, norm_cls(num_channels=module.num_features, **kwargs)
            )
    else:
        raise ValueError(
            f"Invalid normalisation method: {norm_cls!r}. Only {nn.BatchNorm2d}, {nn.GroupNorm!r}, and {nn.LayerNorm!r} are currently supported."
        )

    # Verify that we didn't change any names or add/remove any modules.
    new_named_modules = dict(model.named_modules())
    assert old_named_modules.keys() == new_named_modules.keys()


# def reinit_bn(
#     model: nn.Module,
#     eps: float,
#     momentum: float | None,
#     affine: bool = True,
#     track_running_stats: bool = True,
# ) -> None:
#     for module in model.modules():
#         if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#             module.eps = eps
#             module.momentum = momentum
#             module.affine = affine
#             module.track_running_stats = track_running_stats


def freeze_component(model: nn.Module, name: Literal["encoder", "decoder"]) -> None:
    component: torch.nn.Module = getattr(model, name)
    for param in component.parameters():
        param.requires_grad = False
