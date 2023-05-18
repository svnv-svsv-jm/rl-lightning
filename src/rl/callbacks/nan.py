import torch

from ._base import BaseCallback



class LossNaN(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_before_backward(self, trainer, pl_module, loss, *args):
        if torch.any(torch.isnan(loss)):
            raise ValueError(f"[epoch={trainer.current_epoch}] Loss is NaN for {pl_module.__class__.__name__}. This error was raised by {self.__class__.__name__}")