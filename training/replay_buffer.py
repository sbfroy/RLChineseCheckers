"""
Replay buffer for storing self-play experience.

Stores (spatial, scalars, legal_mask, mcts_policy, value_target) tuples.
Supports random sampling for training batches.
"""

import random
import torch
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class Experience:
    """A single training example from self-play."""
    spatial: np.ndarray      # (C, 17, 17)
    scalars: np.ndarray      # (NUM_SCALAR_FEATURES,)
    legal_mask: np.ndarray   # (ACTION_SPACE_SIZE,) bool
    policy_target: np.ndarray  # (ACTION_SPACE_SIZE,) from MCTS visit counts
    value_target: float        # game outcome from this player's perspective


class ReplayBuffer:
    """
    Fixed-size replay buffer with uniform random sampling.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, exp: Experience):
        """Add a single experience."""
        self.buffer.append(exp)

    def push_batch(self, experiences: List[Experience]):
        """Add a batch of experiences."""
        for exp in experiences:
            self.buffer.append(exp)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch.

        Returns:
            spatial: (B, C, 17, 17)
            scalars: (B, NUM_SCALAR_FEATURES)
            legal_masks: (B, ACTION_SPACE_SIZE)
            policy_targets: (B, ACTION_SPACE_SIZE)
            value_targets: (B, 1)
        """
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

        spatial = torch.from_numpy(np.stack([e.spatial for e in batch]))
        scalars = torch.from_numpy(np.stack([e.scalars for e in batch]))
        legal_masks = torch.from_numpy(np.stack([e.legal_mask for e in batch]))
        policy_targets = torch.from_numpy(np.stack([e.policy_target for e in batch]))
        value_targets = torch.tensor(
            [e.value_target for e in batch], dtype=torch.float32
        ).unsqueeze(1)

        return spatial, scalars, legal_masks, policy_targets, value_targets

    def __len__(self):
        return len(self.buffer)
