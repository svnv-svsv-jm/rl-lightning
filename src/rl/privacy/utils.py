__all__ = ["replace_grucell"]

from loguru import logger

from copy import deepcopy
import torch
from opacus.layers.dp_rnn import DPGRUCell


def replace_grucell(module: torch.nn.Module) -> None:
    """Replaces GRUCell modules with DP-counterparts."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GRUCell) and not isinstance(child, DPGRUCell):
            logger.debug(f"Replacing {name} with {DPGRUCell}")
            replacement = copy_gru(child)
            setattr(module, name, replacement)
    for name, child in module.named_children():
        replace_grucell(child)


def copy_gru(grucell: torch.nn.GRUCell) -> DPGRUCell:
    """Creates a DP-GRUCell from a non-DP one."""
    input_size: int = grucell.input_size
    hidden_size: int = grucell.hidden_size
    bias: bool = grucell.bias
    dpgrucell = DPGRUCell(input_size, hidden_size, bias)
    for name, param in grucell.named_parameters():
        if "ih" in name:
            _set_layer_param(dpgrucell, name, param, "ih")
        elif "hh" in name:
            _set_layer_param(dpgrucell, name, param, "hh")
        else:
            raise AttributeError(f"Unknown parameter {name}")
    return dpgrucell


def _set_layer_param(
    dpgrucell: DPGRUCell,
    name: str,
    param: torch.Tensor,
    layer_name: str,
) -> None:
    """Helper"""
    layer = getattr(dpgrucell, layer_name)
    if "weight" in name:
        layer.weight = torch.nn.Parameter(deepcopy(param))
    elif "bias" in name:
        layer.bias = torch.nn.Parameter(deepcopy(param))
    else:
        raise AttributeError(f"Unknown parameter {name}")
    setattr(dpgrucell, layer_name, layer)
