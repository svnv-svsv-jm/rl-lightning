"""
Base model class. Other models inherit from this class.
"""
import torch
from torch import nn, optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from rl.callbacks import LogWeights, LossNaN
from ._const import LOSS_TRAIN_KEY, LOSS_VAL_KEY, LOSS_TEST_KEY



class Base(pl.LightningModule):
    """Base class for multi-modal models.
    """
    def __init__(
            self,
            lr: float = 1.0e-3,
            patience: int = 2,
            beta1: float = 0.9,
            weight_decay: float = 1.0e-06,
            threshold: float = 3.0e-2,
            cooldown: int = 1,
            every_n_epochs: int = 2,
            example_input_array: torch.Tensor = None,
            # for debugging purposes
            debugging: bool = None,
            **kwargs
        ):
        super().__init__()
        # Set up debug mode
        if debugging is not None:
            self._debugging = debugging
        else:
            self._debugging = False
        # all passed args will be available at `self.hparams` (e.g. argument `opt` will directly be available as `self.hparams.opt`). they will also be passed to MLFlow or other loggers
        self.save_hyperparameters(
            ignore=['example_input_array'],
        )
        # input example for tensorboard graph (don't change attr name)
        self.example_input_array = example_input_array
        # number of optimizers in superclass
        self.n_optims_super = 0


# ------------------
# Config
# ------------------
    def configure_optimizers(self):
        """Define optimizers for this model.
        If you want to use multiple optimizers, the training_step() method should receive the `optimizer_idx` as additional parameter.
        ```
            def training_step(self, batch, batch_idx, optimizer_idx):
                if optimizer_idx == 0:
                    # do training_step with encoder
                    pass
                if optimizer_idx == 1:
                    # do training_step with decoder
                    pass
        ```
        This function can return two lists: one for the optimizers, one for the schedulers. Check PyTorch docs for optimizers and schedulers for more information.
        Example:
        ```
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = SomeScheduler(optimizer1, ...)
                return (
                    {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "your_loss"}},
                    {'optimizer': optimizer2},
                )
        ```
        """
        pass


    def configure_callbacks(self):
        """Configure model-specific callbacks. When the model gets attached, e.g., when .fit() or .test() gets called, the list returned here will be merged with the list of callbacks passed to the Trainer's callbacks argument. If a callback returned here has the same type as one or several callbacks already present in the Trainer's callbacks list, it will take priority and replace them. In addition, Lightning will make sure ModelCheckpoint callbacks run last.
        The string value passed to the key `monitor` must be a key used when the `.log()` or `.log_dict()` methods are called.
        The Lightning checkpoint also saves the arguments passed into the LightningModule init under the `hyper_parameters` key in the checkpoint.
        Returns:
            list: A list of callbacks which will extend the list of callbacks in the Trainer.
        """
        # init list of callbacks
        callbacks = []
        # config earlystopping
        callbacks += [
            EarlyStopping(
                monitor=f"{LOSS_VAL_KEY}",
                patience=3*(self.hparams.patience + 1), # the schedulers will have time to intervene before the training stops
                mode='min', # stops when loss_val stops decreasing
            )
        ]
        # config model checkpoint: you can retrieve the checkpoint after training by calling `checkpoint_callback.best_model_path`
        callbacks += [
            ModelCheckpoint(
                save_last=True,
                monitor=f"{LOSS_VAL_KEY}",
                save_top_k=3,
                mode='min',
                # filename="{epoch:02d}-{LOSS_VAL_KEY:.2f}",
                dirpath=None,
                period=self.hparams.every_n_epochs
            )
        ]
        # learning rate monitor
        callbacks += [LearningRateMonitor(logging_interval='epoch')]
        # gradients
        callbacks += [LogWeights()]
        # NaNs
        callbacks += [LossNaN()]
        return callbacks


# ------------------
# Optim
# ------------------
    def backward(self, loss: torch.Tensor, optimizer, optimizer_idx: int = 0, *args, **kwargs):
        self._debug_info(f"[optimizer_idx:{optimizer_idx}-total:{self.n_optims}-super:{self.n_optims_super}] Calling `loss.backward()`...")
        super().backward(loss, optimizer, optimizer_idx)


    def on_before_zero_grad(self, optimizer):
        self._debug_info(f"Calling `on_before_zero_grad`...")
        super().on_before_zero_grad(optimizer)


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx: int = 0):
        self._debug_info(f"[optimizer_idx:{optimizer_idx}-total:{self.n_optims}-super:{self.n_optims_super}] Calling `optimizer.zero_grad()`...")
        super().optimizer_zero_grad(epoch, batch_idx, optimizer, optimizer_idx)


    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx:int=0, optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None):
        self._debug_info(f"[optimizer_idx:{optimizer_idx}-total:{self.n_optims}-super:{self.n_optims_super}] Calling `optimizer.step()`...")
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)


    def on_before_optimizer_step(self, optimizer, optimizer_idx: int = 0):
        self._debug_info(f"[optimizer_idx:{optimizer_idx}-total:{self.n_optims}-super:{self.n_optims_super}] Calling `on_before_optimizer_step`...")
        super().on_before_optimizer_step(optimizer, optimizer_idx)



# ------------------
# Train
# ------------------
    def on_fit_start(self):
        pass


    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        self._debug_info(f"on_train_batch_start")


    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int = 0):
        """Called for each batch. This is the core of the training procedure.
        Args:
            batch (torch.Tensor):
                Batch of training data. This is an iterable of multi-modal inputs. Check data classes for more info.
            batch_idx (int):
                ID of the training batch.
            optimizer_idx (int):
                ID of the optimizer being used.
        Returns:
            Any of.
                Tensor - The loss tensor
                dict - A dictionary. Can include any keys, but must include the key 'loss'
                None - Training will skip to the next batch
        """
        self._debug_info(f"optimizer_idx: {optimizer_idx}")
        loss = None
        return loss


    def training_step_end(self, step_outputs):
        return step_outputs


    def training_epoch_end(self, step_outputs):
        """Called at the end of the training epoch with the outputs of all training steps. Use this in case you need to do something with all the outputs for every training_step().
        This is currently used to log numerical attributes (but no losses) that may contain practical info to check on a Tensorboard, such as `self.is_ae`, `self.fused`, etc.
        Args:
            training_step_outputs:
                list containing all the outputs of all training steps.
        """
        pass


    def on_epoch_end(self):
        # Information
        attrs = dir(self) # get all attributes' names in a list
        for attr in attrs:
            if attr.startswith('_'):
                # no need to log attrs like '__name__' etc or private ones
                continue
            elif callable(getattr(self, attr)):
                # no need to log methods
                continue
            elif not isinstance(getattr(self, attr), (bool, int, float)):
                # no need to log stuff that is not numerical
                continue
            elif 'loss' in attr:
                # no need to log losses, this is already done elsewhere
                continue
            # log this attr
            self.log(f'info/{attr}', getattr(self, attr))



# ------------------
# Validation
# ------------------
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=None):
        self._debug_info(f"on_validation_batch_start")


    def validation_step_end(self, step_outputs):
        # logging
        self._log_metrics_on_step_end(step_outputs)
        return step_outputs


# ------------------
# Test
# ------------------
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=None):
        self._debug_info(f"on_test_batch_start")


    def test_step_end(self, step_outputs):
        # logging
        self._log_metrics_on_step_end(step_outputs)
        return step_outputs


# ------------------
# private methods
# ------------------
    def _print(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")


    def _debug_info(self, msg: str):
        if self._debugging:
            print(f"[{self.__class__.__name__}:debug-information] {msg}")


    def _log_metrics_on_step_end(self, step_outputs):
        """Useful for `*_step_end()` hooks, which we use for logging.
        Args:
            step_outputs (dict or iterable):
                Object(s) returned by the `*_step()` PyLit hook.
        """
        # error message
        err_msg = f"Input argument `step_outputs` is of type {type(step_outputs)}, which is unsupported and may cause unexepcted behavior. Please make sure that the `*_step_end()` hooks return either an `dict` or an `iterable` of `dict`."
        # logging
        if isinstance(step_outputs, (list, tuple)):
            # log each dict in list
            for metrics in step_outputs:
                assert isinstance(metrics, dict), err_msg
                self._debug_info(f"logging: {metrics}")
                self.log_dict(metrics)
        elif isinstance(step_outputs, dict):
            # log dict directly
            self._debug_info(f"logging: {step_outputs}")
            self.log_dict(step_outputs)
        else:
            raise NotImplementedError(err_msg)


    def _init_weights(self):
        for name, param in self.named_parameters():
            self._print(f"Initializing parameter {name}")
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'conv' in name:
                nn.init.xavier_normal_(param)
            elif 'weight' in name:
                if 'norm' in name:
                    nn.init.ones_(param)
                elif 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    try:
                        nn.init.xavier_normal_(param)
                    except ValueError as ex:
                        nn.init.normal_(param, mean=0., std=1e-1)
            else:
                nn.init.normal_(param, mean=0., std=1e-1)


    def _init_tensors(self):
        self._debug_info(f"Initializing tensors")
        # input / output
        self.real_X = [None] * self.nviews # original inputs
        self.rec_X = [None] * self.nviews # AE-recovered views
        # outputs
        self.Z = [None] * self.nviews  # encoded views
        self.Zg = None  # fused global latent code


    def _add_optimizer(self, params, mode: str, monitor: str, name: str, optimizer_class = optim.Adam):
        """Automates the process of creating an optimizer with corresponding scheduler so that you only have to specify the parameters, the mode for the scheduler with the quantity to monitor, and the name.
        Args:
            params (dict):
                Parameters to track.
            mode (str):
                "min" or "max".
            monitor (str):
                Quantity to monitor for the scheduler.
            name (str):
                Name of the parameter group / learning rate.
        Returns:
            (dict): A dictionary with the needed structure for PyTorch Lightning `configure_optimizers()` hook.
        """
        try:
            optimizer = optimizer_class(
                params,
                lr=float(self.hparams.lr),
                betas=(float(self.hparams.beta1), 0.999),
                weight_decay=float(self.hparams.weight_decay)
            )
        except ValueError as ex:
            raise Exception(f"Error encountered for optimizer {name}. Info:\n params= {params}") from ex
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=.1,
            mode=mode,
            patience=self.hparams.patience,
            threshold=self.hparams.threshold,
            cooldown=self.hparams.cooldown,
        )
        self._debug_info(f"Configured optimizer {name}:\n{optimizer}\nWith scheduler (monitoring {monitor}):\n{scheduler}\n")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
                "name": f'lr/{name}',
            }
        }


    def _optim_name(self, *args, **kwargs):
        name = self._get_optim_name(*args, **kwargs)
        self._debug_info(f"Optimizer name: {name}")
        return name


    def _get_optim_name(self, idx: int = None):
        if idx is None:
            return [self._strip_optim_name(opt) for opt in self.optims]
        else:
            return self._strip_optim_name(self.optims[idx])


    def _strip_optim_name(self, opt: dict):
        name = opt.get('lr_scheduler').get('name')
        start = name.find('lr/') + len('lr/')
        name = name[start:]
        return name
