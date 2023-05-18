__all__ = ["HydraConfigCallback"]

import typing as ty
from loguru import logger
import pprint

from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import parsing
from omegaconf import DictConfig, OmegaConf


class HydraConfigCallback(pl.Callback):
    """Logs Hydra configuration to the logger."""

    def __init__(self, cfg: DictConfig, clean_namespace: bool = True) -> None:
        """
        Args:
            cfg (DictConfig):
                Hydra configuration.
            clean_namespace (bool, optional):
                Whether to clean the namespace. Defaults to True.
        """
        super().__init__()
        config = OmegaConf.to_container(cfg)
        assert isinstance(config, dict), f"{type(config)}"
        if clean_namespace:
            parsing.clean_namespace(config)
        to_log = pprint.pformat(config, indent=2)
        logger.debug(f"Config: {to_log}")
        self.cfg = config.copy()

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str = None,
    ) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        # Return if no loggers
        if not hasattr(trainer, "loggers"):
            logger.warning("No loggers to log to.")
            return
        # Make sure we have a list of loggers
        loggers = trainer.loggers
        if not isinstance(loggers, ty.Iterable):
            loggers = [loggers]
        # For each logger, log
        for pl_logger in loggers:
            logger.debug(f"Logging to {type(pl_logger)}...")
            pl_logger.log_hyperparams(self.cfg)
            # If TensorBoard, log the YAML config as text
            if isinstance(pl_logger, TensorBoardLogger):
                writer: SummaryWriter = pl_logger.experiment
                # Log using OmegaConf
                yaml_data = OmegaConf.to_yaml(self.cfg)
                yaml_data = "".join("\t" + line for line in yaml_data.splitlines(True))
                step = trainer.global_step
                writer.add_text("config", yaml_data, step)  # type: ignore
