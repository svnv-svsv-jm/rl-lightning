__all__ = ["DPOptimizer"]

import typing as ty
from opacus.optimizers import DPOptimizer as OpacusDPOptimizer
import torch


def params(
    optimizer: torch.optim.Optimizer,
    accepted_names: ty.List[str] = None,
) -> ty.List[torch.nn.Parameter]:
    """
    Return all parameters controlled by the optimizer
    Args:
        accepted_names (ty.List[str]):
            List of parameter group names you want to apply DP to. This allows you to choose to apply DP only to specific parameter groups. Of course, this will work only if the optimizer has named parameter groups. IF it doesn't, then this argument will be ignored and DP will be applied to all parameter groups.
    Returns:
        (ty.List[torch.nn.Parameter]): Flat list of parameters from all `param_groups`
    """
    # lower case
    if accepted_names is not None:
        accepted_names = [name.lower() for name in accepted_names]
    # unwrap parameters from the param_groups into a flat list
    ret = []
    for param_group in optimizer.param_groups:
        if accepted_names is not None and "name" in param_group.keys():
            name: str = param_group["name"].lower()
            if name.lower() in accepted_names:
                ret += [p for p in param_group["params"] if p.requires_grad]
        else:
            ret += [p for p in param_group["params"] if p.requires_grad]
    return ret


class DPOptimizer(OpacusDPOptimizer):
    """Brainiac-2's DP-Optimizer"""

    def __init__(
        self,
        *args: ty.Any,
        param_group_names: ty.List[str] = None,
        **kwargs: ty.Any,
    ) -> None:
        """Constructor."""
        self.param_group_names = param_group_names
        super().__init__(*args, **kwargs)

    @property
    def params(self) -> ty.List[torch.nn.Parameter]:
        """
        Returns a flat list of ``nn.Parameter`` managed by the optimizer
        """
        return params(self, self.param_group_names)

    def clip_and_accumulate(self) -> ty.Any:
        """Performs gradient clipping. Stores clipped and aggregated gradients into `p.summed_grad`."""
        return super().clip_and_accumulate()
