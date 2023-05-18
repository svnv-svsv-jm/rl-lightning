__all__ = ["GATEncoder"]

from loguru import logger
import typing as ty
from typing import Dict, Optional

from torch import nn, Tensor
from torch_geometric.nn.conv import GATConv


class GATEncoder(nn.Module):
    """GAT Encoder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        num_layers: int = 1,
        hidden_size: int = 256,
    ) -> None:
        """_summary_
        Args:
            in_dim (int): _description_
            out_dim (int): _description_
        """
        super().__init__()
        heads = int(heads)
        num_layers = int(num_layers)
        hidden_dims = num_layers * [hidden_size]
        layers = [ConvSkipLayer(in_dim, hidden_dims[0], heads=heads)]
        for h in hidden_dims:
            layers.append(ConvSkipLayer(heads * h, h, heads=heads))
        layers.append(ConvSkipLayer(heads * hidden_dims[-1], out_dim, heads=heads))
        self.conv_skip = nn.ModuleList(layers)
        logger.trace(f"Initialized {self.__class__.__name__}")

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edges.
        Returns:
            Tensor: Embedding.
        """
        logger.trace(f"X: {x.size()}, edge_index: {edge_index.size()}")
        for layer in self.conv_skip:
            x = layer(x, edge_index)
        return x


class ConvSkipLayer(nn.Module):
    """Conv + Skip operation."""

    def __init__(self, in_channels: int, out_channels: int, heads: int) -> None:
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.skip = nn.Linear(in_channels, heads * out_channels)
        self.elu = nn.ELU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edges.
        Returns:
            Tensor: Embedding.
        """
        logger.trace(f"X: {x.size()}, edge_index: {edge_index.size()}")
        x = self.conv(x, edge_index) + self.skip(x)
        logger.trace(f"X: {x.size()}")
        x = self.elu(x)
        return x
