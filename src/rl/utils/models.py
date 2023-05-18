__all__ = ["dynamic_setup", "setup_embedder"]

from loguru import logger
import typing as ty
from typing import Union, Sequence, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
import pytorch_lightning as pl

from . import class_for_name
from .data import get_data_from_trainer

LOADER_TYPE = Union[
    DataLoader[Any],
    Sequence[DataLoader[Any]],
    Sequence[Sequence[DataLoader[Any]]],
    Sequence[Dict[str, DataLoader[Any]]],
    Dict[str, DataLoader[Any]],
    Dict[str, Dict[str, DataLoader[Any]]],
    Dict[str, Sequence[DataLoader[Any]]],
]


def setup_embedder(
    embedder: torch.nn.Module = None,
    ckpt_path: str = None,
    strict: bool = True,
    load_from_checkpoint_kwargs: ty.Dict[str, ty.Any] = {},
    model_class: ty.Union[ty.Type[pl.LightningModule], str] = None,
) -> torch.nn.Module:
    """Set up embedder."""
    # If no ckpt_path, return with input model
    if ckpt_path is None:
        assert isinstance(
            embedder, torch.nn.Module
        ), "Provide an embedder when not providing a checkpoint path."
        return embedder
    # We have to load model from checkpoint: get the loading fn
    if model_class is not None:
        if isinstance(model_class, str):
            model_class = class_for_name(model_class)
        load = model_class.load_from_checkpoint
    else:
        assert isinstance(embedder, pl.LightningModule)
        load = embedder.load_from_checkpoint
    # Load the model
    embedder = load(ckpt_path, strict=strict, **load_from_checkpoint_kwargs)
    # Return
    logger.debug(f"Imported embedder from: {ckpt_path}")
    return embedder.eval()


def dynamic_setup(trainer: pl.Trainer) -> ty.Tuple[int, int]:
    """Understands in_features and number of classes from the trainer. This can be used to dynamically set up a pytorch_lightning model during the `trainer.fit()` call."""
    # Find training dataloader
    dataloader: LOADER_TYPE
    loaders = get_data_from_trainer(trainer)
    for loader in loaders:
        if loader is not None:
            dataloader = loader
            break
    if loader is None:
        dataloader = trainer._data_connector._train_dataloader_source.dataloader()
    assert isinstance(
        dataloader, DataLoader
    ), f"Unsupported dynamic setup for when a training dataloader is not of type {DataLoader}. Got it of type {type(dataloader)}."
    # Get information from batch
    for batch in dataloader:
        # Make sure we have a Data object in the loader so that we can get the number of node features
        assert isinstance(
            batch, Data
        ), f"Dynamic setup is possible only for dataloaders/datamodules that return {Data} objects. Received batch of type {type(batch)}."
        x: torch.Tensor = batch.x
        in_features = int(x.size(1))
        # We can infer the number of classes only if there is a datamodule of type LightningDataset
        out_features: int
        if hasattr(trainer, "datamodule"):
            dm = trainer.datamodule  # type: ignore
            if hasattr(dm, "dataset") and isinstance(dm.dataset, Dataset):
                out_features = dm.dataset.num_classes
            else:
                logger.warning(
                    f"Cannot infer the number of classes (setting `out_features` to 0): unsupported dynamic setup for datamodule of type {type(dm)}. To enable this, use a datamodule that has an attribute 'dataset' of type {Dataset}. Ignore this warning if your model does not need the number of classes."
                )
                out_features = 0
        else:
            raise AttributeError(
                "Trainer has no datamodule we can retrieve the original dataset from. Dynamic setup not possible."
            )
        break
    return in_features, out_features
