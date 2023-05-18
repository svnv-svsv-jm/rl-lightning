__all__ = ["augment", "bernoulli_mask", "barlow_twins_loss"]

from typing import Union, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data


def augment(
    data: Data,
    p_x: float,
    p_e: float,
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Args:
        data (Data): Graph to augment.
        p_x (float): Node masking probability.
        p_e (float): Edge dropping probability.
    Returns:
        Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
            Two graphs, each represented by a Tuple of node features and edges.
    """
    device = data.x.device
    x = data.x
    num_fts = x.size(-1)
    ei = data.edge_index
    num_edges = ei.size(-1)
    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.0]
    ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.0]
    return (x_a, ei_a), (x_b, ei_b)


def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float) -> Tensor:
    """
    Args:
        size (Union[int, Tuple[int, ...]]): _description_
        prob (float): _description_
    Returns:
        Tensor: Probabilities.
    """
    return torch.bernoulli((1 - prob) * torch.ones(size))


def barlow_twins_loss(z_a: torch.Tensor, z_b: torch.Tensor, r_tol: float = 1e-9) -> torch.Tensor:
    """
    Args:
        z_a (torch.Tensor): Embeddings of graph A.
        z_b (torch.Tensor): Embeddings of graph B.
    Returns:
        torch.Tensor: Loss.
    """
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim
    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + r_tol)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + r_tol)
    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size
    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss: torch.Tensor
    loss = (1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(2).sum()
    return loss
