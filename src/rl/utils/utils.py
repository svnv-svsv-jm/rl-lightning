__all__ = [
    "type_check",
    "pretty_dict",
    "override_default_cfg",
    "delete_files_in_directory",
    "check_for_nan",
    "class_for_name",
    "save_artifacts",
]

import typing as ty
from copy import deepcopy
import os, glob, importlib, yaml
from pathlib import Path
import torch


def save_artifacts(
    data: dict,
    name: str,
    artifact_location: str = None,
) -> None:
    """Save artifacts."""
    if artifact_location is None:
        return
    Path(artifact_location).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(artifact_location, f"{name}.yaml")
    with open(filename, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, indent=2)


def class_for_name(name: str) -> ty.Type:
    """Load a class from string indicating module's name and the class name.
    Args:
        name (str):
            The path to the class to load. Example: rl.query.Query.
    Returns:
        A class.
    """
    class_name = name.split(".")[-1]
    # load the module, will raise ImportError if module cannot be loaded
    module_name = name.split(".")[0]
    for x in name.split(".")[1:-1]:
        module_name += f".{x}"
    module = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    klass = getattr(module, class_name)
    return klass  # type: ignore


def check_for_nan(x: torch.Tensor, extra: str = None) -> None:
    """Raises error if Nan."""
    if torch.isnan(x).any():
        msg = f"NaN's: {x}\t"
        if extra is not None:
            msg += extra
        raise ValueError(msg)


def delete_files_in_directory(folder: str, wildcard: str = "*") -> None:
    """Deletes all files in a directory."""
    files = glob.glob(os.path.join(folder, wildcard))
    for file in files:
        os.remove(file)


def override_default_cfg(cfg: dict, cfg_default: dict) -> dict:
    """Override default configuration: only from the provided keys.
    Args:
        cfg (dict):
            Configuration to use. Only the present keys will be used.
        cfg_default (dict):
            Default configuration. Keys not present in `cfg` will be given the value from this configuration.
    """
    cfg_out = deepcopy(cfg_default)
    for key, val in cfg.items():
        cfg_out[key] = val
    return cfg_out


def pretty_dict(dictionary: dict, indent: int = 0) -> str:
    """Pretty doctionary stringifier."""
    pretty_string = ""
    for key, value in dictionary.items():
        pretty_string += "\t" * indent + str(key) + "\n"
        if isinstance(value, dict):
            pretty_string += pretty_dict(value, indent + 1)
        else:
            pretty_string += "\t" * (indent + 1) + str(value) + "\n"
    return pretty_string


def type_check(variable: ty.Any, typing: ty.Any, can_be_none: bool = False) -> None:
    """Checks type is ok."""
    if can_be_none:
        if variable is None:
            return
    assert isinstance(
        variable,
        typing,
    ), f"variable must be of type {typing} but was found of type {type(variable)}"
