import pytorch_lightning as pl

from ._base import BaseCallback



class FreezeOptimizer(BaseCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def on_train_epoch_end(self, trainer, pl_module, *args):
        if trainer.current_epoch == 10:
            pl_module.optimizer.freeze()