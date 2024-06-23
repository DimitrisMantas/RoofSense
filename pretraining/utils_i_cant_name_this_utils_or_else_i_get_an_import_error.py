from collections import OrderedDict

import torch
from torch import Tensor


def get_encoder_params(ckpt_path: str) -> tuple[str, OrderedDict[str, Tensor]]:
    ckpt = torch.load(ckpt_path)

    name: str
    if "backbone" in ckpt["hyper_parameters"]:
        name = ckpt["hyper_parameters"]["backbone"]
    elif "encoder" in ckpt["hyper_parameters"]:
        name = ckpt["hyper_parameters"]["encoder"]
    else:
        msg = (
            "Failed to locate a hyperparameter key corresponding to the encoder name. "
            "Supported parameter names are 'backbone' and 'encoder'."
        )
        raise RuntimeError(msg)

    identifier="encoder."

    weights: OrderedDict[str, Tensor] = ckpt["state_dict"]
    weights = OrderedDict(
        {name: value for name, value in weights.items() if identifier in name}
    )
    weights = OrderedDict(
        {name.replace(identifier,""): value for name, value in weights.items()}
    )

    return name, weights
