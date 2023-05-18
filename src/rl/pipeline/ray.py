__all__ = ["ray_tune"]

from loguru import logger
import typing as ty

import os
import hydra
from omegaconf import DictConfig, OmegaConf

try:
    HASRAY = True
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
except ImportError:
    HASRAY = False

from .runner import run


def train_fn(config: dict = {}, cfg: DictConfig = DictConfig({})) -> None:
    """Training function to be called by Ray.
    Args:
        config (dict):
            Search space. Ray passes one instance of the search space, which then needs to overwrite the configuration.
        cfg (DictConfig):
            Config for the run.
    """
    # Update cfg with sampled search space
    if config:
        for key, value in config.items():
            OmegaConf.update(cfg, key, value, force_add=True)
    # Run pipeline
    run(cfg)


def ray_tune(cfg: DictConfig, train_fn: ty.Callable = train_fn) -> None:
    """Runs `ray.tune.run()` with additional options.
    Args:
        cfg (DictConfig):
            Loaded YAML Hydra configuration.
        train_fn (ty.Callable):
            Function that runs the training pipeline.
    """
    if not HASRAY:
        raise RuntimeError("Please install ray to run this: `pip install ray`")
    logger.info(f"Local directory: {cfg.paths.output_dir}")
    # Sanity checks
    assert "ray" in cfg, "No Ray configuration found."
    # Get the additional config for tune.run
    tune_run: dict = OmegaConf.to_container(hydra.utils.instantiate(cfg.ray.tune_run))  # type: ignore
    # Create scheduler
    kwargs: dict = OmegaConf.to_container(hydra.utils.instantiate(cfg.ray.ray_scheduler))  # type: ignore
    scheduler = ASHAScheduler(**kwargs)
    # Create search space
    search_space: ty.Optional[dict]
    parameter_columns: ty.Optional[ty.List[str]]
    if "search_space" in cfg:
        search_space = hydra.utils.instantiate(cfg.search_space)
        assert isinstance(search_space, dict)
        parameter_columns = list(search_space.keys())
    else:
        tune_run["search_alg"] = None
        search_space = None
        parameter_columns = None
    # CLIReporter
    reporter = CLIReporter(parameter_columns=parameter_columns)
    # Prepare training function with parameters
    train_fn_with_parameters = tune.with_parameters(train_fn, cfg=cfg)  # type: ignore
    # Run and get final analysis
    analysis = tune.run(
        train_fn_with_parameters,
        config=search_space,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=f"{cfg.paths.output_dir}",
        **tune_run,
    )
    # Save best configuration to YAML
    best_config: DictConfig = analysis.best_config
    logger.info(f"Best hyperparameters found were ({type(best_config)}): {best_config}")
    name = tune_run.get("name", "ray")
    filename = os.path.join(f"{cfg.paths.output_dir}", name, "best-config.yaml")
    OmegaConf.save(best_config, filename)
