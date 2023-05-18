from typing import Tuple
import torch
from torch import nn, Tensor



def dqn_mse_loss(
        net: nn.Module,
        target_net: nn.Module,
        batch: Tuple[Tensor, Tensor],
        gamma: float,
    ) -> Tensor:
    """Calculates the mse loss using a mini batch from the replay buffer.
    Args:
        net: network
        target_net: target network
        batch: current mini batch of replay data
    Returns:
        loss
    """
    states, actions, rewards, dones, next_states = batch
    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()
    expected_state_action_values = rewards + gamma*next_state_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)