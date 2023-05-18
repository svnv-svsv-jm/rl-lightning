# pylint: disable=unused-import
__all__ = ["register_custom_resolvers"]

from loguru import logger
from typing import Callable, Any, Optional
from functools import wraps
from omegaconf import OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

from .model import (
    ResolveModel,
    ResolveParams,
    ResolveMetric,
    resolve_model,
    resolve_params,
)
from .datamodule import ResolveDatamodule, resolve_datamodule


def register_custom_resolvers(
    version_base: Optional[str],
    config_path: Any,
    config_name: str,
    clear: bool = True,
) -> Callable:
    """Optional decorator to register custom OmegaConf resolvers. It is expected to be call before `hydra.main` decorator call.
    Returns:
        Callable: Decorator that registers custom resolvers before running main function.
    """
    config_path = f"{config_path}"
    logger.trace(f"{version_base}, {config_path}, {config_name}")
    # Sneak peek into the config to read important params
    with hydra.initialize_config_dir(
        config_dir=config_path,
        version_base=version_base,
    ):
        cfg = hydra.compose(
            config_name=config_name,
            return_hydra_config=True,
            overrides=[],
        )
        logger.debug(cfg)
    if clear:
        GlobalHydra.instance().clear()
    # Register resolvers with read params
    OmegaConf.register_new_resolver(
        "oc.resolve_datamodule", ResolveDatamodule(cfg.inputs.model), replace=True
    )
    OmegaConf.register_new_resolver(
        "oc.resolve_model", ResolveModel(cfg.inputs.model), replace=True
    )
    OmegaConf.register_new_resolver(
        "oc.resolve_params", ResolveParams(cfg.inputs.model), replace=True
    )
    OmegaConf.register_new_resolver(
        "oc.resolve_metric", ResolveMetric(cfg.inputs.model), replace=True
    )
    # Define decorator
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapper

    return decorator
