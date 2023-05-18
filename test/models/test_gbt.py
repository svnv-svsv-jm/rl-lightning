import pytest
import sys, os
from loguru import logger

from torch_geometric.data import HeteroData
from torch_geometric.datasets import PPI, IMDB
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningNodeData
import pytorch_lightning as pl

from rl.models import GBT


def test_gbt_on_ppi(data_path: str) -> None:
    """Test on dataset used in original paper."""
    logger.info("Testing on dataset used in original paper.")
    # Get data
    loader_train = _loader(data_path, "train")
    loader_val = _loader(data_path, "val")
    # Trainer
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, max_steps=4)
    # Model
    feature_dim = int(loader_train.dataset[0].x.size(1))
    latent_dim = 4
    model = GBT(feature_dim, latent_dim)
    # Train
    trainer.fit(model, loader_train)
    trainer.test(model, loader_val)
    logger.success("OK.")


def test_gbt_on_imdb(data_path: str) -> None:
    """Test it can be trained on IMDB."""
    logger.info("Testing on IMDB.")
    dataset = IMDB(data_path)
    data: HeteroData = dataset[0]
    dm = LightningNodeData(data.to_homogeneous(), num_neighbors=2 * [20])
    # Trainer
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, max_steps=4, devices=1)
    # Model
    latent_dim = 4
    model = GBT(latent_dim=latent_dim)
    # Train
    trainer.fit(model, dm)
    # Encode a graph
    for graph in dm.train_dataloader():
        model(graph)
        break
    logger.success("OK.")


def _loader(data_path: str, split: str) -> DataLoader:
    """Helper."""
    logger.info(f"Getting loader for {split}")
    dataset = PPI(root=os.path.join(data_path, "PPI", split), split=split)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint", "--all"])
