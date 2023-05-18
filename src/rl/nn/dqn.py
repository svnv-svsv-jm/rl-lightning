from torch import nn

from ._base import Base
from .mlp import MLP



class DQN(Base):
    """Simple MLP network."""
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = MLP(
            dims=[obs_size, hidden_size, n_actions],
            activation=nn.ReLU,
            with_last_layer_activation=False,
        )

    def forward(self, x):
        return self.net(x.float())