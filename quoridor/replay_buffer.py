"""Fixed-size replay buffer of (planes, policy_target, value_target) samples.

Supports recency-weighted sampling so recent data dominates batches,
preventing old stale samples from pulling the value head in wrong directions.
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, recent_ratio=0.5):
        self.buffer = deque(maxlen=capacity)
        self.recent_ratio = recent_ratio
        # Track how many samples have been added so we can split old vs new
        self._total_added = 0
        # When we add a batch, we record (start_idx, end_idx) for each chunk
        # so we know which samples are from the most recent iteration.
        self._chunk_bounds = []  # list of (start, end) in buffer insertion order

    def __len__(self):
        return len(self.buffer)

    def add_many(self, samples):
        n = len(samples)
        if n == 0:
            return
        start = self._total_added
        self._total_added += n
        self.buffer.extend(samples)
        self._chunk_bounds.append((start, start + n))
        # Drop chunk bounds that have fallen out of the buffer
        oldest_kept = self._total_added - len(self.buffer)
        self._chunk_bounds = [
            (s, e) for s, e in self._chunk_bounds if e > oldest_kept
        ]

    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0:
            raise ValueError("buffer is empty")
        k = min(batch_size, n)

        if self.recent_ratio <= 0 or not self._chunk_bounds:
            # Uniform sampling
            indices = random.sample(range(n), k)
        else:
            # Split: recent_ratio fraction from the most recent chunk,
            # the rest uniformly from the whole buffer.
            n_recent = max(1, int(k * self.recent_ratio))
            n_uniform = k - n_recent

            # Most recent chunk maps to the tail of the buffer
            last_start, last_end = self._chunk_bounds[-1]
            # Convert from global insertion indices to buffer indices
            offset = self._total_added - n
            buf_start = max(0, last_start - offset)
            buf_end = min(n, last_end - offset)
            if buf_end <= buf_start:
                # Recent chunk has been fully evicted; fall back to uniform
                indices = random.sample(range(n), k)
            else:
                recent_indices = random.choices(
                    range(buf_start, buf_end), k=n_recent
                )
                uniform_indices = random.sample(range(n), n_uniform)
                indices = recent_indices + uniform_indices

        planes = np.stack([self.buffer[i][0] for i in indices]).astype(np.float32)
        pi = np.stack([self.buffer[i][1] for i in indices]).astype(np.float32)
        z = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        return planes, pi, z
