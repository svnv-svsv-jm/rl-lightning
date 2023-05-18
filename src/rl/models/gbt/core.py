__all__ = ["GBT"]

from typing import Any
from loguru import logger
import typing as ty

import torch
from torch_geometric.data import Data
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from rl.utils import dynamic_setup
from .nn import GATEncoder
from .utils import barlow_twins_loss, augment


class GBT(pl.LightningModule):
    """Graph Barlow Twins model from: https://arxiv.org/pdf/2106.02466.pdf
    Inspired by https://github.com/pbielak/graph-barlow-twins/blob/master/gssl/inductive/model.py
    """

    def __init__(
        self,
        in_features: int = None,
        latent_dim: int = None,
        p_x: float = 0.1,
        p_e: float = 0.1,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-5,
        checkpoint_callback: bool = False,
        checkpoint_kwargs: ty.Dict[str, ty.Any] = {},
        **kwargs: ty.Any,
    ) -> None:
        """_summary_
        Args:
            in_features (int):
                Node feature dimension.
            latent_dim (int):
                Latent representation dimension.
            p_x (float):
                Probability of masking node features.
            p_e (float):
                Probability of edge masking.
            learning_rate (float):
                Learning rate.
            weight_decay (float):
                Weight decay.
            checkpoint_callback (bool, optional):
                Whether to use the CheckpointCallback.
            **kwargs:
                See :class:`GATEncoder`.
        """
        super().__init__()
        self.save_hyperparameters()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.p_x = p_x
        self.p_e = p_e
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_kwargs = checkpoint_kwargs.copy()
        # Encoder
        if in_features is not None and self.latent_dim is not None:
            self.encoder = GATEncoder(in_dim=in_features, out_dim=self.latent_dim, **kwargs)

    def setup(self, stage: str = None) -> None:
        """Set up dynamically."""
        if stage is not None and (stage.lower() == "fit") and hasattr(self, "trainer"):
            self.in_features, _ = dynamic_setup(self.trainer)
            assert isinstance(self.in_features, int)
            assert self.latent_dim is not None
            self.encoder = GATEncoder(in_dim=self.in_features, out_dim=int(self.latent_dim))

    def configure_optimizers(self) -> ty.Dict[str, ty.Any]:
        """Optimization."""
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "frequency": 1,
        }

    def configure_callbacks(self) -> ty.List[pl.Callback]:  # type: ignore
        """Configure checkpoint."""
        callbacks = []
        if self.checkpoint_callback:
            self.checkpoint_kwargs.setdefault("monitor", "loss/train")
            self.checkpoint_kwargs.setdefault("mode", "min")
            self.checkpoint_kwargs.setdefault("save_top_k", 3)
            self.checkpoint_kwargs.setdefault("save_last", True)
            self.checkpoint_kwargs.setdefault("save_on_train_epoch_end", True)
            logger.info(f"Creating {cb.ModelCheckpoint} with parameters: {self.checkpoint_kwargs}")
            ckpt_cb_val: pl.Callback = cb.ModelCheckpoint(**self.checkpoint_kwargs)
            callbacks.append(ckpt_cb_val)
        return callbacks

    def forward(self, batch: Data) -> torch.Tensor:
        """Forward."""
        h: torch.Tensor = self.encoder(x=batch.x, edge_index=batch.edge_index)
        return h

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "test")

    def step(self, batch: Data, batch_idx: int, stage: str) -> torch.Tensor:
        """_summary_
        Args:
            batch (Data): _description_
            batch_idx (int): _description_
        Returns:
            torch.Tensor: Loss.
        """
        logger.trace(f"batch_idx={batch_idx}")
        (x_a, ei_a), (x_b, ei_b) = augment(data=batch, p_x=self.p_x, p_e=self.p_e)
        z_a = self.encoder(x=x_a, edge_index=ei_a)
        z_b = self.encoder(x=x_b, edge_index=ei_b)
        loss: torch.Tensor = barlow_twins_loss(z_a=z_a, z_b=z_b)
        self.log(f"loss/{stage}", loss)
        return loss
