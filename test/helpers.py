__all__ = ["set_cfg_defaults"]

import pytest
import typing as ty
from loguru import logger

import pyrootutils
from omegaconf import DictConfig, open_dict
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm


def set_cfg_defaults(cfg: DictConfig) -> DictConfig:
    """Set defaults for all tests."""
    root_dir = pyrootutils.find_root(indicator=[".git", "setup.py"])
    if root_dir is None:
        root_dir = "."
    with open_dict(cfg):
        cfg.paths.root_dir = root_dir
        cfg.trainer.max_steps = 2
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.num_processes = 1
        cfg.trainer.fast_dev_run = False
        cfg.trainer.enable_checkpointing = False
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.stage = "fit"
    return cfg
