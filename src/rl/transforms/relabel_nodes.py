__all__ = ["RelabelNodes"]

from torch import Tensor
from torch_geometric.data import Data
import torch_geometric.transforms as T


class RelabelNodes(T.BaseTransform):
    """Relabel graph nodes by offsetting."""

    def __call__(self, data: Data) -> Data:
        """Transformation happens here."""
        y: Tensor = data.y
        offset = y.min()
        y = y - offset
        data.y = y
        return data
