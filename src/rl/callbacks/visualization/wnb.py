import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from rl.callbacks._base import BaseCallback
from rl.callbacks.utils import find_logger



class LogWeights(BaseCallback):
    """Logs model's learnable weights to TensorBoard.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_steps = 0


    def on_init_start(self, trainer, *args):
        self._n_steps = 0


    def on_before_zero_grad(self, trainer, pl_module, outputs):
        # Handle logger
        logger = find_logger(pl_module, TensorBoardLogger)
        if logger is None:
            self._debug_info("No logger was found.")
            return
        # Avoid logging all the time
        self._n_steps += 1
        if self._n_steps % self.log_every_n_steps != 0:
            return
        # Log
        for name, param in pl_module.named_parameters():
            if param is not None and param.requires_grad:
                self._debug_info(f"Logging {name}")
                # log parameter
                logger.experiment.add_histogram(
                    f"{name}/value",
                    param,
                    self._n_steps,
                )
                # log gradient
                if param.grad is not None:
                    logger.experiment.add_histogram(
                        f"{name}/grad",
                        param.grad,
                        self._n_steps,
                    )
            # else:
            #     self._print(f"Parameter {name} of model {pl_module.__class__.__name__} has gradient {param.grad}.")
