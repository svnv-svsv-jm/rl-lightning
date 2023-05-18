from collections import OrderedDict
from typing import List, Tuple
import torch
from torch import Tensor, nn, optim
from torch.optim import Optimizer

from rl.nn import DQN as DQNnet
from rl.agent import Agent
from rl.losses import dqn_mse_loss
from ._base import Base
from ._const import LOSS_TRAIN_KEY, LOSS_VAL_KEY, LOSS_TEST_KEY


class DQN(Base):
    """Basic DQN Model."""

    def __init__(
        self,
        obs_size: int = None,
        n_actions: int = None,
        datamodule=None,
        gamma: float = 0.99,
        sync_rate: int = 10,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        agent: Agent = None,
        **kwargs,
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
        """
        super().__init__(**kwargs)
        if obs_size is None and n_actions is None:
            assert (
                datamodule is not None
            ), f"Please provide a datamodule or specify obs_size and n_actions."
            if datamodule.fit is None:
                datamodule.setup(stage="fit")
            obs_size = datamodule.fit.env.observation_space.shape[0]
            n_actions = datamodule.fit.env.action_space.n
        self.save_hyperparameters(ignore=["dm"])
        self.net = DQNnet(obs_size, n_actions)
        self.target_net = DQNnet(obs_size, n_actions)
        self.total_reward = 0
        self.episode_reward = 0
        self.agent = agent
        self._is_buffer_populated = False

    # ------------------
    # Custom methods
    # ------------------
    def populate(self, agent: Agent, net: nn.Module, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with experiences.
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            agent.play_step(net, epsilon=1.0)

    def _loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        return dqn_mse_loss(
            net=self.net,
            target_net=self.target_net,
            batch=batch,
            gamma=self.hparams.gamma,
        )

    # ----------------------
    # Validation and test
    # ----------------------
    def on_validation_start(self):
        # Check if agent is present
        if self.agent is None:
            self.agent = self.trainer.datamodule.fit.agent

    def on_test_start(self):
        # Check if agent is present
        if self.agent is None:
            self.agent = self.trainer.datamodule.test.agent

    # ------------------
    # Forward
    # ------------------
    def forward(self, X: Tensor) -> Tensor:
        """Passes in a state X through the network and gets the q_values of each action as an output.
        Args:
            X: environment state
        Returns:
            Q-values
        """
        q_values = self.net(X)
        return q_values

    # ------------------
    # Training methods
    # ------------------
    def on_fit_start(self):
        super().on_fit_start()
        # Check if agent is present
        if self.agent is None:
            self.agent = self.trainer.datamodule.fit.agent
        # Populate buffer on start
        if not self._is_buffer_populated:
            self.populate(
                self.agent,
                self.net,
                self.hparams.warm_start_steps,
            )

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        optimizer_idx: int = 0,
    ) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        """
        loss = super().training_step(batch, batch_idx, optimizer_idx)
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        )
        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        # calculates training loss
        loss = self._loss(batch)
        if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        # Logging
        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        # Progress bar
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }
        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        self._debug_info("Configuring optimizers...")
        # learning rate
        lr = float(self.hparams.lr)
        # collect parameters
        params = {
            "params": self.net.parameters(),
            "name": f"{self.net.__class__.__name__}",
            "lr": lr,
        }
        # init
        optims = []
        # optimizer
        optimizer = self._add_optimizer(
            params,
            mode="min",
            monitor=f"{LOSS_VAL_KEY}",
            name=self.__class__.__name__,
            optimizer_class=optim.Adam,
        )
        optims.append(optimizer)
        assert isinstance(
            optims, (tuple, list)
        ), f"ImplementationError: this method must return a list or subclasses will not be able to manipulate its output and append any other optimizers."
        self.n_optims = len(optims)
        self.optims = optims  # override
        return optims

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
