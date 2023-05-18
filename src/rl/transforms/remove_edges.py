__all__ = ["RemoveEdges"]

from loguru import logger
import typing as ty

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from rl.utils.data import _sample_n_elements


class RemoveEdges(BaseTransform):
    """Transforms a graph by removing edges."""

    def __init__(
        self,
        n_edges: int = None,
        p_edges: float = None,
        edges_idx: ty.List[int] = None,
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            n_edges (int, optional):
                Number of edges to be removed. Defaults to None.
            p_edges (float, optional):
                Percentage of edges to be removed. Defaults to None.
            edges_idx (ty.List[int], optional):
                ID of edges to be removed. Defaults to None.
        """
        super().__init__(**kwargs)
        self.p_edges = p_edges
        self.n_edges = n_edges
        self.edges_idx = edges_idx

    def __call__(self, data: Data) -> Data:
        """Transformation happens here."""
        edge_index = data.edge_index
        assert isinstance(edge_index, torch.Tensor)
        n_edges_total = int(edge_index.size(1))
        if self.edges_idx is not None:
            sampled_edges = torch.Tensor(self.edges_idx).long()
        elif self.n_edges is not None:
            sampled_edges, _ = _sample_n_elements(n_edges_total, n_members=self.n_edges)
        elif self.p_edges is not None:
            sampled_edges, _ = _sample_n_elements(n_edges_total, prior=self.p_edges)
        else:
            raise RuntimeError(
                "Please provide `p_edges`, `n_edges` or `edges_idx` when creating an instance of this class."
            )
        data.edge_index = self._rm_edge(edge_index, sampled_edges)
        return data

    def _rm_edge(self, edge_index: torch.Tensor, edges_idx: torch.Tensor) -> torch.Tensor:
        """Removes index."""
        n_edges_total = int(edge_index.size(1))
        logger.debug(f"n_edges_total={(n_edges_total)}")
        mask = torch.ones(n_edges_total).bool().to(edge_index.device)
        edges_idx = edges_idx.view(-1).long()
        for i in edges_idx:
            mask[i] = False
        edge_index = edge_index[:, mask]
        return edge_index
