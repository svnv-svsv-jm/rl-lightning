import torch
import pytorch_lightning as pl

from rl.utils import Timer
from rl.utils import print_debug



class BaseCallback(pl.callbacks.Callback):
    """Base class for rl's callbacks.
    """
    # Constructor
    def __init__(
            self,
            opt: dict = None,
            verbose: bool = False,
            log_every_n_steps: int = 1000,
            on_train: bool = False,
            **kwargs
        ):
        super().__init__()
        self.opt = opt
        self._verbose = verbose
        self._error_raised = False
        self._warning = False
        self._datasets = ('train', 'val', 'test')
        self.log_every_n_steps = log_every_n_steps
        self.on_train = on_train


    # --------------------------
    # Exception handling
    # --------------------------
    def on_exception(self, trainer, pl_module, exception):
        pass


    # ------------------
    # Private methods
    # ------------------
    def _print(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")


    def _raise(self, ex: Exception, msg: str):
        raise ex(f"[{self.__class__.__name__}] {msg}")


    def _error_msg(self,
            batch_idx: int,
            current_epoch: int,
            ex: Exception,
            msg: str = None,
        ):
        err_msg = f"Error for batch {batch_idx} at epoch {current_epoch}: {ex}"
        if msg is not None:
            err_msg += f"\n\t{msg}"
        if not self._error_raised:
            self._error_raised = True
            self._print(err_msg)


    def _debug_info(self, msg: str):
        if self._verbose: self._print(msg)