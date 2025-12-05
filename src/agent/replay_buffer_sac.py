# replay_buffer_sac.py
import random
from collections import deque
from typing import Deque, Tuple, NamedTuple, Sequence

import numpy as np
import torch


class Transition(NamedTuple):
    """
    A single experience tuple used for SAC training.

    Each field corresponds to:
        - state:       observation before the action
        - action:      action taken by the agent
        - reward:      scalar reward received
        - next_state:  observation after the action
        - done:        whether the episode ended (1.0) or not (0.0)
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBufferContinuous:
    """
    A fixed-size replay buffer for storing transitions used by SAC.

    The buffer stores Transition objects and supports uniform random sampling.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """
        Store a new transition in the replay buffer.
        """
        self.buffer.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Uniformly sample a batch of transitions and convert them to torch tensors.

        Returns:
            states, actions, rewards, next_states, dones
        """
        # batch is a list of Transition objects
        batch: Sequence[Transition] = random.sample(self.buffer, batch_size)

        # "Transpose" the batch: group all states, all actions, etc.
        batch_t: Transition = Transition(*zip(*batch))

        states = torch.tensor(
            np.stack(batch_t.state), dtype=torch.float32
        )
        actions = torch.tensor(
            np.stack(batch_t.action), dtype=torch.float32
        )
        rewards = torch.tensor(
            batch_t.reward, dtype=torch.float32
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.stack(batch_t.next_state), dtype=torch.float32
        )
        dones = torch.tensor(
            batch_t.done, dtype=torch.float32
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Returns the number of stored transitions.
        """
        return len(self.buffer)