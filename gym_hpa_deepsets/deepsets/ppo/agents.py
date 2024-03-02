from typing import Optional

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

"""
Here the deepsets wiring is implemented, as shown in the deepsets paper. 
Only change is the inner layers, from 64 to 128, and the input's shape.
"""

def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class EquivariantLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.Gamma = nn.Linear(in_channels, out_channels, bias=False)
        self.Lambda = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, n_elements, in_channels)
        # return: (batch_size, n_elements)
        xm, _ = torch.max(x, dim=1, keepdim=True)
        return self.Lambda(x) - self.Gamma(xm)


class EquivariantDeepSet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            EquivariantLayer(in_channels, hidden_channels),
            nn.ReLU(),
            EquivariantLayer(hidden_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_elements, in_channels)
        # return: (batch_size, n_elements)
        return torch.squeeze(self.net(x), dim=-1)


class InvariantDeepSet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.psi = nn.Sequential(
            EquivariantLayer(in_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, hidden_channels),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_elements, in_channels)
        # return: (batch_size, n_elements)
        x = torch.mean(self.psi(x), dim=1)
        return torch.squeeze(self.rho(x), dim=-1)


class DeepSetAgent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv) -> None:
        super().__init__()
        in_channels = ((envs.observation_space.shape[0]) // envs.get_attr('num_apps')[0]) + 1 # Input's shape

        # Actor outputs pi(a|s)
        self.actor = EquivariantDeepSet(in_channels)

        # Critic outputs V(s)
        self.critic = InvariantDeepSet(in_channels)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, deterministic: bool = True) -> torch.Tensor:
        """
        It gets the action from the agent based on an input. Used during testing.
        :param x: The tensor input
        :param masks: masking (optional)
        :param deterministic: It indicates if tha action will always be of the highest probability or a just a sample
        :return: the logit's number, which is the action number
        """
        logits = self.actor(x)
        if masks is not None:
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype)
            logits = torch.where(masks, logits, HUGE_NEG)
        dist = Categorical(logits=logits)
        if deterministic:
            return dist.mode
        return dist.sample()

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None, masks: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the action as well as the critic's evaluation of the action from the agent. Used during training.
        :param x: The tensor input
        :param action: The action to be made (in training it is None)
        :param masks: Masking (optional
        :return: The action number, the , the calculated entropy and the critic's evaluation.
        """

        # Load the logits of the actor.
        logits = self.actor(x)
        if masks is not None:
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype)
            logits = torch.where(masks, logits, HUGE_NEG)
        # The Categorical function converts the logits from meaningless numbers to probabilities
        dist = Categorical(logits=logits)
        if action is None:
            # Take a sample of the distribution. It take the probabilities of each action in consideration.
            action = dist.sample()
            # This print statement indicates the actions that the agent chose at each time.
            # It is used to be able to see the algorithm's preference of actions during training
            print(action)
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)
