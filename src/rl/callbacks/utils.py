# pylint: disable=broad-except
__all__ = ["find_logger"]

from loguru import logger
import typing as ty

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger


def find_logger(
    this: ty.Union[pl.LightningModule, pl.Trainer],
    logger_type: ty.Type,
) -> ty.Optional[Logger]:
    """Checks if there is a logger of the type specified by the argument `logger`. For example:
    >>> isinstance(logger, (TensorBoardLogger, CSVLogger, MLFlowLogger))
    Args:
        this (pl.LightningModule | pl.Trainer):
            Model or Trainer.
        logger_type (Logger):
            A Pytorch Lightning logger class.
    """
    name = "pl_module" if isinstance(this, pl.LightningModule) else "trainer"
    # Try here
    if hasattr(this, "loggers") and isinstance(this.loggers, ty.Iterable):
        logger.trace(f"Looking for {logger_type} in {this.loggers}")
        for log in this.loggers:
            if isinstance(log, logger_type):
                assert isinstance(log, Logger)
                return log
    logger.trace(f"No logger at {name}.loggers.")
    # And here
    if hasattr(this, "logger"):
        logger.trace(f"Looking for {logger_type} in {this.logger}")
        if isinstance(this.logger, logger_type):
            assert isinstance(this.logger, Logger)
            return this.logger
    logger.trace(f"No logger at {name}.logger.")
    return None
