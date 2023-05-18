import gym
from typing import Tuple
from torch import nn
from torch.utils.data.dataset import IterableDataset

from rl.buffer import ReplayBuffer
from rl.agent import Agent
from rl.utils import class_for_name



class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.
    """
    def __init__(self,
            replay_size: int = 1000,
            sample_size: int = 16,
            env: str = "CartPole-v0",
            warm_start_steps: int = 1000,
            agent_class: str = Agent,
            **kwargs
        ) -> None:
        self.sample_size = sample_size
        self.buffer = ReplayBuffer(replay_size)
        self.env = gym.make(env)
        if isinstance(agent_class, str):
            agent_class = class_for_name(agent_class, "agent")
        self.agent = agent_class(self.env, self.buffer)

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]