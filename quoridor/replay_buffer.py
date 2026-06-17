"""Fixed-size replay buffer of (planes, policy_target, value_target) samples."""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add_many(self, samples):
        self.buffer.extend(samples)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        planes = np.stack([b[0] for b in batch]).astype(np.float32)
        pi = np.stack([b[1] for b in batch]).astype(np.float32)
        z = np.array([b[2] for b in batch], dtype=np.float32)
        return planes, pi, z
