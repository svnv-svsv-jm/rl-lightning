import gym
from typing import Tuple
import torch
from torch import nn
import numpy as np

from rl.buffer import ReplayBuffer
from rl.buffer import Experience



class Agent:
    """Base Agent class handling the interaction with the environment."""
    def __init__(self,
            env: gym.Env,
            replay_buffer: ReplayBuffer
        ) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.
        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])
            if device not in ["cpu"]:
                state = state.cuda(device)
            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
        return action

    @torch.no_grad()
    def play_step(self,
            net: nn.Module,
            epsilon: float = 0.0,
            device: str = "cpu",
        ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.
        Args:
            net: DQN-like network
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """
        action = self.get_action(net, epsilon, device)
        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)
        self.state = new_state
        if done:
            self.reset()
        return reward, done