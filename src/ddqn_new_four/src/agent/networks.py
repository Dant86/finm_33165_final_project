import torch
import torch.nn as nn
from typing import Tuple


class QNetwork(nn.Module):
    """
    Fully-connected neural network used as the Q-value function approximator
    in the DDQN agent.

    Input:
        - state_dim: dimension of the observation vector
    Output:
        - Q-values for each discrete action (size = action_dim)

    Architecture:
        A simple 3-layer MLP with ReLU nonlinearity.
        This is typically sufficient for portfolio management tasks where
        the state dimension is moderate.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network.

        Args:
            x: input tensor of shape (batch_size, state_dim)

        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.model(x)
