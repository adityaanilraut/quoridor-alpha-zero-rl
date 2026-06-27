"""Agents that select an action for the player to move in a real game state."""

import numpy as np

from . import minimax
from .game import flip_action
from .mcts import MCTS


class RandomAgent:
    name = "random"

    def select_action(self, state):
        return int(np.random.choice(state.legal_actions()))


class MinimaxAgent:
    def __init__(self, depth=2, max_walls=12):
        self.depth = depth
        self.max_walls = max_walls
        self.name = f"minimax(d={depth})"

    def select_action(self, state):
        return minimax.best_action(state, self.depth, self.max_walls)


class NeuralMCTSAgent:
    """Plays with the trained net + MCTS. ``temperature`` 0 means greedy."""

    def __init__(self, net, config, temperature=0.0, add_noise=False):
        self.net = net
        self.config = config
        self.mcts = MCTS(net, config)
        self.temperature = temperature
        self.add_noise = add_noise
        self.name = "alphazero"

    def select_action(self, state):
        canonical = state.canonical()
        if getattr(self.config, "use_gumbel", True):
            pi, _ = self.mcts.run_gumbel(canonical)
        else:
            counts = self.mcts.run_batched(canonical, add_noise=self.add_noise)
            total = counts.sum()
            pi = counts / total if total > 0 else canonical.legal_actions_mask()
        if self.temperature <= 1e-6:
            a_can = int(np.argmax(pi))
        else:
            probs = pi ** (1.0 / self.temperature)
            probs = probs / probs.sum()
            a_can = int(np.random.choice(len(probs), p=probs))
        # the policy is in the canonical (player-0) frame; map back to absolute
        return a_can if state.current_player == 0 else flip_action(a_can, state.N)
