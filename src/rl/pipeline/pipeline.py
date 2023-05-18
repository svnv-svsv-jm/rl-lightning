__all__ = ["Pipeline"]

import typing as ty
from loguru import logger

import torch
import pytorch_lightning as pl


from rl.callbacks import DifferentialPrivacy
from rl.utils import override_default_cfg, wrong_type_err_msg
from .utils import DEFAULT_TRAINER_KWARGS, SUPPORTED_STAGES


_OUT_METRIC = ty.Union[torch.Tensor, ty.Dict[str, torch.Tensor]]
_OUT_DICT = ty.Dict[str, _OUT_METRIC]


class Pipeline:
    """Pipeline."""

    def __init__(
        self,
        model: torch.nn.Module = None,
        datamodule: pl.LightningDataModule = None,
        trainer: pl.Trainer = None,
        # privacy kwargs
        dp: bool = False,  # pylint: disable=invalid-name
        dp_kwargs: dict = {},
        # optimize
        optimize_metric: str = None,
    ) -> None:
        """Pipeline for training, validation, test. This will essentially be like doing the following:

        >>> model: pl.LightningModule = MyModel()
        >>> datamodule: pl.LightningDataModule = MyData()
        >>> trainer = pl.Trainer(...)
        >>> trainer.fit(model, datamodule)
        >>> trainer.validate(model, datamodule)
        >>> ...

        With some additional functionalities, like enabling Differential Privacy (DP) and perform a privacy risk assessment at the end of the training.
        Args:
            model (torch.nn.Module):
                A model to train. If a `torch.nn.Module`, no training will be possible.
            datamodule (pl.LightningDataModule):
                Datamodule for the training pipeline.
            dp (bool, optional): Defaults to False.
                Whether to enable DP or not.
            privacy_assess (bool, optional):
                Whether to perform privacy risk assessment or not.
            query (Query, optional):
                WQuery for privacy assessment.
            trainer (pl.Trainer):
                Lightning trainer.
            mace_kwargs (dict):
                Kwargs for MACE.
            dp_kwargs (dict):
                Kwargs for differential privacy.
            should_attack (bool):
                Whether to attack the target model.
            attack_type (str):
                Attack instance.
        """
        # sanity checks
        assert isinstance(model, torch.nn.Module)
        assert isinstance(datamodule, pl.LightningDataModule)
        # inputs
        self.model = model
        self.datamodule = datamodule
        self.dp = dp  # pylint: disable=invalid-name
        if optimize_metric is None:
            optimize_metric = "loss/val"
        self.optimize_metric = optimize_metric
        # attributes
        self.dp_cb: ty.Optional[DifferentialPrivacy] = None
        self.dp_kwargs: dict = dp_kwargs.copy()
        self.trainer: pl.Trainer
        if trainer is not None:
            self.config_trainer(trainer)
        self.val_metrics: ty.Optional[ty.Sequence[dict]] = None
        self.test_metrics: ty.Optional[ty.Sequence[dict]] = None
        # init datamodule
        try:
            self.datamodule.prepare_data()
            self.datamodule.setup("fit")
        except AttributeError as ex:
            logger.debug(
                f"Could not init datamodule, probably because there is nothing to set up with the provided one. The error message was: {ex}"
            )

    def enable_dp(self, **dp_kwargs: ty.Any) -> None:
        """Enable DP.
        Args:
            **dp_kwargs (Any):
                See :class:`~rl.callbacks.DifferentialPrivacy`
        """
        if dp_kwargs:
            self.dp_kwargs = dp_kwargs
        logger.debug(
            f"Enabling Differential Privacy for {self.model.__class__.__name__} with {self.dp_kwargs}"
        )
        self.dp_cb = DifferentialPrivacy(**self.dp_kwargs)

    def config_trainer(
        self,
        trainer: pl.Trainer = None,
        **trainer_kwargs: ty.Any,
    ) -> None:
        """Configure Trainer. You can provide a trainer object or set up a trainer here using kwargs.
        Args:
            trainer (pl.Trainer):
                Trainer object. Defaults to None.
            **trainer_kwargs (Any):
                See :class:`~pytorch_lightning.Trainer`.
        """
        if isinstance(trainer, pl.Trainer):
            self.trainer = trainer
        elif trainer_kwargs:
            trainer_kwargs = override_default_cfg(
                trainer_kwargs,
                DEFAULT_TRAINER_KWARGS,
            )
            self.trainer = pl.Trainer(**trainer_kwargs)
        elif not hasattr(self, "trainer") or self.trainer is None:
            self.trainer = pl.Trainer(**DEFAULT_TRAINER_KWARGS)  # type: ignore
        else:
            assert hasattr(self, "trainer")
            assert isinstance(self.trainer, pl.Trainer), wrong_type_err_msg(
                self.trainer, pl.Trainer
            )

    def run(
        self,
        stage: str = "fit",
        trainer: pl.Trainer = None,
        **trainer_kwargs: ty.Any,
    ) -> None:
        """Runs pipeline.
        Args:
            stage (str, optional): Defaults to "fit".
                One of 'fit', 'validate', 'test' or 'assess'.
            trainer (pl.Trainer, optional): Defaults to None.
                `pl.Trainer` to train/validate/test the model.
        Raises:
            ValueError: if unrecognized `stage` is passed.
        """
        self.setup(stage, trainer, **trainer_kwargs)
        # call appropriate stage: see SUPPORTED_STAGES
        stage = stage.lower()
        if stage == "fit":
            self.fit()
        elif stage == "validate":
            self.validate()
        elif stage == "test":
            self.test()
        # end
        logger.info("+++ Job has finished +++")

    def setup(
        self,
        stage: str,
        trainer: pl.Trainer = None,
        **trainer_kwargs: ty.Any,
    ) -> None:
        """Set up all components."""
        # check stage exists
        if stage not in SUPPORTED_STAGES:
            raise ValueError(
                f"Unsupported stage {stage}. Supported values are: {SUPPORTED_STAGES}."
            )
        # trainer
        self.config_trainer(trainer, **trainer_kwargs)
        # dp
        if self.dp:
            self.enable_dp()
        if self.dp_cb is not None:
            self.trainer.callbacks.append(self.dp_cb)  # type: ignore

    def fit(self) -> None:
        """Runs the training loop."""
        if isinstance(self.model, pl.LightningModule):
            logger.debug(f"Fitting {self.model.__class__.__name__}")
            self.trainer.fit(self.model, self.datamodule)

    def validate(self) -> ty.Sequence[dict]:
        """Runs validation loop."""
        assert isinstance(self.model, pl.LightningModule), self._assert_is_pl_module_msg()
        self.val_metrics = self.trainer.validate(self.model, self.datamodule)
        return self.val_metrics

    def test(self) -> ty.Sequence[dict]:
        """Runs validation loop."""
        assert isinstance(self.model, pl.LightningModule), self._assert_is_pl_module_msg()
        self.test_metrics = self.trainer.test(self.model, self.datamodule)
        return self.test_metrics

    def fast_dev_run(self) -> None:
        """Fast dev run."""
        assert isinstance(self.model, pl.LightningModule), self._assert_is_pl_module_msg()
        pl.Trainer(fast_dev_run=True).fit(self.model, self.datamodule)

    def summary(self) -> dict:
        """Summary."""
        summary = {"Test metrics": self.test_metrics}
        return summary

    def logged_metrics(self) -> ty.Optional[_OUT_DICT]:
        """Tries to retrieve logged metrics from the trainer. Optionally looks for a MetricTrackerCallback callback and returns its logged metrics."""
        try:
            logged_metrics = self.trainer.logged_metrics
        except Exception:
            return None
        return logged_metrics

    def get_metric_to_optimize(self) -> ty.Optional[float]:
        """For Optuna."""
        # Get stuff logged during training / validation / test
        logged_metrics = self.logged_metrics()
        if logged_metrics is None:
            return None
        # Now get the metric we want to run HPO for
        if isinstance(self.optimize_metric, str):
            # If one metric, just get it
            key = self.optimize_metric
            out = self._get_metric(key, logged_metrics)
            return out
        # If multiple metrics, then return average among them
        outs = []
        for key in self.optimize_metric:
            outs.append(self._get_metric(key, logged_metrics))
        return sum(outs) / len(outs)

    def _get_metric(self, key: str, logged_metrics: dict) -> float:
        """Get one metric."""
        metric: _OUT_METRIC = logged_metrics[key]
        # Ok if numerical
        if isinstance(metric, (int, float)):
            return float(metric)
        # If Tensor, convert to float
        assert isinstance(
            metric, torch.Tensor
        ), f"This is currently only supported for Tensor metrics but got {metric}."
        output = float(metric.mean())
        return output

    def _assert_is_pl_module_msg(self) -> str:
        """Assertion error message."""
        return f"Model is of type {type(self.model)} but should be of type {pl.LightningModule}."
