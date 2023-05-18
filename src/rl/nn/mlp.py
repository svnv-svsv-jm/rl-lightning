from typing import List
from torch import nn, Tensor

from ._base import Base



class MLP(Base):
    def __init__(self,
            dims: List[int],
            activation = nn.ReLU,
            with_last_layer_activation: bool = True,
            last_act = nn.LogSoftmax(1),
            bias: bool = True,
            check_dims: bool = False,
        ):
        """MLP class.
        Args:
            dims (list):
                Contains each dimension, from input to output including each hidden layer. The total number of layers will be `len(dims) - 1`.
            activation (optional):
                Class of the activation function to use. Defaults to nn.ReLU.
            with_last_layer_activation (bool):
                Whether to use last layer activation function. Defaults to True
            last_act (optional):
                Class of the last activation function to use. Defaults to nn.Tanh.
            check_dims (bool):
                Enforce decreasing layer dimensions. Defaults to False.
        """
        super().__init__()
        if check_dims:
            dims = self._check_dims(*dims)
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = (ind == (len(dims_pairs) - 1))
            linear = nn.Linear(
                int(dim_in),
                int(dim_out),
                bias=bias
            )
            layers.append(linear)
            if not is_last:
                layers.append(activation())
            elif with_last_layer_activation:
                try:
                    layers.append(last_act)
                except Exception as ex:
                    layers.append(last_act())
        self.mlp = nn.Sequential(*layers)


    def forward(self, X: Tensor):
        return self.mlp(X)


    def _check_dims(self,
            *dims: int,
        ):
        """Checks that layer hidden dimensions decrease from `dims[0]` to `dims[-1]`. If not, it enforces it.
        Args:
            *dims (int):
                Layer sizes.
        """
        # iterate over dims
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        output = [dims[0]]
        for i, (dim_in, dim_out) in enumerate(dims_pairs):
            if dim_out <= dim_in:
                output.append(dim_out)
        # last dimension
        dim_out = dims[-1]
        if len(output) == 1:
            self._warn("MLP dimensions are wrongly configured. Output dimension > Input dimension.")
            output.append(dim_out)
        elif output[-1] != dim_out:
            self._warn("MLP dimensions are wrongly configured. Output dimension > Previous dimensions (hidden layers).")
            output[-1] = dim_out
        return output

    def _print(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

    def _warn(self, msg):
        print(f"[{self.__class__.__name__}:WARNING] {msg}")