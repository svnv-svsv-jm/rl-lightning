__all__ = ["DifferentialPrivacy"]

from loguru import logger
import typing as ty

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus import GradSampleModule

from rl.privacy import DPOptimizer, replace_grucell


class DifferentialPrivacy(pl.callbacks.EarlyStopping):
    """Enables differential privacy using Opacus. Converts optimizers to instances of the :class:`~opacus.optimizers.DPOptimizer` class. This callback inherits from `EarlyStopping`, thus it is also able to stop the training when enough privacy budget has been spent. Please beware that Opacus does not support multi-optimizer training.
    For more info, check the following links:
    * https://opacus.ai/tutorials/
    * https://blog.openmined.org/differentially-private-deep-learning-using-opacus-in-20-lines-of-code/
    """

    def __init__(
        self,
        budget: float = 1.0,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = None,
        use_target_values: bool = False,
        idx: ty.Sequence[int] = None,
        log_spent_budget_as: str = "DP/spent-budget",
        param_group_names: ty.List[str] = None,
        private_dataloader: bool = False,
        default_alphas: ty.Sequence[ty.Union[float, int]] = None,
        **gsm_kwargs: ty.Any,
    ) -> None:
        """Enables differential privacy using Opacus. Converts optimizers to instances of the :class:`~opacus.optimizers.DPOptimizer` class. This callback inherits from `EarlyStopping`, thus it is also able to stop the training when enough privacy budget has been spent.
        Args:
            budget (float, optional): Defaults to 1.0.
                Maximun privacy budget to spend.
            noise_multiplier (float, optional): Defaults to 1.0.
                Noise multiplier.
            max_grad_norm (float, optional): Defaults to 1.0.
                Max grad norm used for gradient clipping.
            delta (float, optional): Defaults to None.
                The target δ of the (ϵ,δ)-differential privacy guarantee. Generally, it should be set to be less than the inverse of the size of the training dataset. If `None`, this will be set to the inverse of the size of the training dataset `N`: `1/N`.
            use_target_values (bool, optional):
                Whether to call `privacy_engine.make_private_with_epsilon()` or `privacy_engine.make_private`. If `True`, the value of `noise_multiplier` will be calibrated automatically so that the desired privacy budget will be reached only at the end of the training.
            idx (ty.Sequence[int]):
                List of optimizer ID's to make private. Useful when a model may have more than one optimizer. By default, all optimizers are made private.
            log_spent_budget_as (str, optional):
                How to log and expose the spent budget value. Although this callback already allows you to stop the training when enough privacy budget has been spent (see argument `stop_on_budget`), this keyword argument can be used in combination with an `EarlyStopping` callback, so that the latter may use this value to stop the training when enough budget has been spent.
            param_group_names (ty.List[str]):
                List of parameter group names you want to apply DP to. This allows you to choose to apply DP only to specific parameter groups. Of course, this will work only if the optimizer has named parameter groups. If it doesn't, then this argument will be ignored and DP will be applied to all parameter groups.
            private_dataloader (bool, optional):
                Whether to make the dataloader private. Defaults to False.
            **gsm_kwargs:
                Input arguments for the :class:`~opacus.GradSampleModule` class.
        """
        # inputs
        self.budget = budget
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.use_target_values = use_target_values
        self.log_spent_budget_as = log_spent_budget_as
        self.param_group_names = param_group_names
        self.private_dataloader = private_dataloader
        self.gsm_kwargs = gsm_kwargs
        if default_alphas is None:
            self.default_alphas = RDPAccountant.DEFAULT_ALPHAS + list(range(64, 150))
        else:
            self.default_alphas = default_alphas
        # init early stopping callback
        super().__init__(
            monitor=self.log_spent_budget_as,
            mode="max",
            stopping_threshold=self.budget,
            check_on_train_epoch_end=True,
            # we do not want to stop if spent budget does not increase. this may even be desirable
            min_delta=0,
            patience=1000000,
        )
        # attributes
        self.epsilon: float = 0.0
        self.best_alpha: float = 0.0
        self.accountant = RDPAccountant()
        self.idx = idx  # optims to privatize

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str = None,
    ) -> None:
        """Call the GradSampleModule() wrapper to add attributes to pl_module."""
        if stage == "fit":
            replace_grucell(pl_module)
            try:
                pl_module = GradSampleModule(pl_module, **self.gsm_kwargs)
            except ImportError as ex:
                raise ImportError(
                    f"{ex}. This may be due to a mismatch between Opacus and PyTorch version."
                ) from ex

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when the training epoch begins. Use this to make optimizers private."""
        # idx
        if self.idx is None:
            self.idx = range(len(trainer.optimizers))
        # dp dataloader
        dp_dataloader = None
        if self.private_dataloader:
            if isinstance(trainer.train_dataloader, DataLoader):
                dataloader = trainer.train_dataloader
                dp_dataloader = DPDataLoader.from_data_loader(dataloader, distributed=False)
                trainer.train_dataloader = dp_dataloader
            elif isinstance(trainer.train_dataloader.loaders, DataLoader):  # type: ignore
                dataloader = trainer.train_dataloader.loaders  # type: ignore
                dp_dataloader = DPDataLoader.from_data_loader(dataloader, distributed=False)
                trainer.train_dataloader.loaders = dp_dataloader  # type: ignore
            else:
                logger.debug("No dataloader found.")
        # get dataloader
        if dp_dataloader is not None:
            data_loader = dp_dataloader
        else:
            data_loader = trainer._data_connector._train_dataloader_source.dataloader()
        sample_rate: float = 1 / len(data_loader)
        dataset_size: int = len(data_loader.dataset)  # type: ignore
        expected_batch_size = int(dataset_size * sample_rate)
        # delta
        if self.delta is None:
            self.delta = 1 / dataset_size
        # make optimizers private
        optimizers: ty.List[Optimizer] = []
        dp_optimizer: ty.Union[Optimizer, DPOptimizer]
        for i, optimizer in enumerate(trainer.optimizers):
            if not isinstance(optimizer, DPOptimizer) and i in self.idx:
                if self.use_target_values:
                    self.noise_multiplier = get_noise_multiplier(
                        target_epsilon=self.budget / 2,
                        target_delta=self.delta,
                        sample_rate=sample_rate,
                        epochs=trainer.max_epochs,
                        accountant="rdp",
                    )
                dp_optimizer = DPOptimizer(
                    optimizer=optimizer,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                    expected_batch_size=expected_batch_size,
                    param_group_names=self.param_group_names,
                )
                dp_optimizer.attach_step_hook(
                    self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
                )
            else:
                dp_optimizer = optimizer
            optimizers.append(dp_optimizer)
        trainer.optimizers = optimizers

    def on_train_batch_end(  # pylint: disable=unused-argument # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: ty.Any,
        batch: ty.Any,
        batch_idx: int,
        *args: ty.Any,
    ) -> None:
        """Called after the batched has been digested. Use this to understand whether to stop or not."""
        self._log_and_stop_criterion(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run at the end of the training epoch."""
        logger.debug(
            f"Spent budget (epoch={trainer.current_epoch}): {self.epsilon}. Max budget: {self.budget}."
        )

    def get_privacy_spent(self) -> ty.Tuple[float, float]:
        """Estimate spent budget."""
        # get privacy budget spent so far
        epsilon, best_alpha = self.accountant.get_privacy_spent(
            delta=self.delta,
            alphas=self.default_alphas,
        )
        return float(epsilon), float(best_alpha)

    def _log_and_stop_criterion(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Logging privacy spent: (epsilon, delta) and stopping if necessary."""
        self.epsilon, self.best_alpha = self.get_privacy_spent()
        # log: expose the spent budget, an EarlyStopping callback may use this value to stop the training when enough budget has been spent
        pl_module.log(
            self.log_spent_budget_as,
            self.epsilon,
            on_epoch=True,
            prog_bar=True,
        )
        if self.epsilon > self.budget:
            logger.info(
                f"The training will stop at epoch {trainer.current_epoch} and step {trainer.global_step} because all the allowed privacy budget ({self.budget}) has been spent: {self.epsilon}."
            )
            trainer.should_stop = True
