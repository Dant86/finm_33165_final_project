import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import QNetwork
from .replay_buffer import ReplayBuffer


class DDQNAgent:
    """
    Double Deep Q-Network (DDQN) Agent.

    Supports:
        - epsilon-greedy exploration
        - online and target Q-networks
        - MSE loss on Bellman targets
        - soft or hard target network updates
        - experience replay buffer

    This implementation is specifically tuned for portfolio management tasks
    but is general enough to be used for other discrete-action RL problems.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        tau: float = 0.01,
        target_update_freq: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            state_dim: dimension of the observation vector
            action_dim: number of discrete actions
            hidden_dim: size of MLP hidden layers
            gamma: discount factor
            lr: learning rate
            buffer_size: capacity of replay buffer
            batch_size: batch size for updates
            tau: soft update coefficient (if using soft updates)
            target_update_freq: if set, perform hard update every N steps
            device: "cpu" or "cuda", autodetected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # Online and target networks
        self.online_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    # ---------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon: choose random action.
        Otherwise: choose action with highest predicted Q-value.
        """
        if random.random() < epsilon:
            return random.randrange(self.online_net.model[-1].out_features)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ---------------------------------------------------------
    # Learning Update
    # ---------------------------------------------------------
    def update(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform one DDQN update step using a batch from the replay buffer.
        If not enough transitions are available, returns (None, None).
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample replay batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_values = self.online_net(states).gather(1, actions)

        # Double DQN target
        with torch.no_grad():
            # Action selection via online network
            next_q_online = self.online_net(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Q-values via target network
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions)

            # Bellman target
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # Compute TD loss
        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step_counter += 1

        # Target network update
        if self.target_update_freq is None:
            # Soft update every step
            with torch.no_grad():
                for target_param, param in zip(
                    self.target_net.parameters(), self.online_net.parameters()
                ):
                    target_param.data.mul_(1.0 - self.tau)
                    target_param.data.add_(self.tau * param.data)
        else:
            # Hard update every N steps
            if self.learn_step_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

        # Diagnostics: average max Q-value
        with torch.no_grad():
            max_q_vals = self.online_net(states).max(dim=1)[0]
            avg_max_q = max_q_vals.mean().item()

        return float(loss.item()), float(avg_max_q)
