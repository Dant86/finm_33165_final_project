import random
from collections import deque, namedtuple
from typing import Deque, Tuple

import numpy as np
import torch


# A single transition in the environment
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done"),
)


class ReplayBuffer:
    """
    Standard experience replay buffer for off-policy RL algorithms.

    Stores tuples of:
        (state, action, reward, next_state, done)

    Supports:
        - push(): add a transition
        - sample(batch_size): randomly sample a batch
        - __len__(): buffer size

    Implementation uses a fixed-size deque for efficiency and simplicity.
    """

    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """Store a single transition in the replay buffer."""
        self.buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly sample a batch of transitions and convert them into
        PyTorch tensors ready for the DDQN update step.
        """
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))  # transpose

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32)
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(
            np.stack(batch.next_state), dtype=torch.float32
        )
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
