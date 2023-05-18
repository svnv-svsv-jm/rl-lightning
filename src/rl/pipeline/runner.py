# pylint: disable=abstract-class-instantiated
__all__ = ["run"]

from loguru import logger

import sys, os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from rl.utils import Logger
from rl.pipeline import Pipeline
from rl.pipeline.utils import setup_from_cfg


def run(cfg: DictConfig) -> Pipeline:
    """Runs the main pipeline of Brainiac-2.
    Args:
        cfg (DictConfig):
            Hydra configuration.
    Returns:
        Pipeline:
            Brainiac-2's Pipeline object.
    """
    # Logging stuff
    log_level = cfg.log_level if "log_level" in cfg else "INFO"
    logger.add(sys.stderr, level=log_level.upper())
    # get job logging file from hydra config and add it as sink for loguru and also redirect both stdout and stderr to it
    run_loc = None
    try:
        hydra_cfg = HydraConfig.get()
        job_log_file: str = hydra_cfg.job_logging["handlers"].file.filename
        run_loc = os.path.join(os.path.dirname(job_log_file), "artifacts")
        logger.info(f"Logging redirected to: {job_log_file}")
        logger.add(job_log_file, level="INFO")
        log_redirect = Logger(filename=job_log_file)
        sys.stdout = log_redirect  # type: ignore
        sys.stderr = log_redirect  # type: ignore
    except Exception:
        logger.warning("Could not redirect logging to file.")
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}\n")
    # create model, data and trainer
    pl_model, pl_datamodule, pl_trainer, stage = setup_from_cfg(cfg)
    # optimize metric
    optimize_metric = cfg.get("optimize_metric", None)
    # create pipeline
    pipeline = Pipeline(
        pl_model,
        pl_datamodule,
        trainer=pl_trainer,
        optimize_metric=optimize_metric,
    )
    # Fast dev run to check for errors
    pipeline.fast_dev_run()
    # call trainer with appropriate stage method
    logger.info(f"Running location: {run_loc}")
    pipeline.run(stage, artifact_location=run_loc)
    return pipeline
