__all__ = ["ToHomogeneous"]

from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import BaseTransform

from rl.utils import wrong_type_err


class ToHomogeneous(BaseTransform):
    """Transforms an heterogeneous graph to a homogeneous one."""

    def __call__(self, data: HeteroData) -> Data:
        """Transformation happens here."""
        # Do nothing if already homo
        if not isinstance(data, HeteroData):
            wrong_type_err(data, Data)
            return data
        # Tranform to homo data
        homogeneous_data = data.to_homogeneous()
        return homogeneous_data
