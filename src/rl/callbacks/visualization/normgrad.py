import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from rl.callbacks._base import BaseCallback
from rl.callbacks.utils import find_logger



class NormGrad(BaseCallback):
    """Logs the total gradients for each backprop. This will take a lot of storage.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        # Handle logger
        logger = find_logger(pl_module, TensorBoardLogger)
        if logger is None:
            return
        # Avoid logging all the time
        self._n_steps += 1
        if self._n_steps % self.log_every_n_steps != 0:
            return
        # Estimate total gradient and log
        attr = "global_step"
        if hasattr(pl_module, attr):
            # Total grad
            total_grad = 0.
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    total_grad = total_grad + param.grad.norm()
            # Logging
            logger.experiment.add_scalar(
                'total_grad',
                total_grad,
                getattr(pl_module, attr),
            )
        elif not self._warning:
            self._print(f"Model {pl_module.__class__.__name__} has no attribute {attr}.")
            self._warning = True