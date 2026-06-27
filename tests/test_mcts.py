"""Gumbel MuZero search checks. Run: python -m pytest tests/test_mcts.py"""

import numpy as np

from quoridor.config import get_config
from quoridor.game import QuoridorState, action_space_size, OFFSET_TO_INDEX
from quoridor.mcts import MCTS


class _DummyNet:
    """Uniform-prior, zero-value net so the search drives the result."""

    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, planes):
        return np.zeros(self.action_size, dtype=np.float32), 0.0

    def predict_states(self, states):
        n = len(states)
        return (np.zeros((n, self.action_size), dtype=np.float32),
                np.zeros(n, dtype=np.float32))


def _mcts(cfg):
    return MCTS(_DummyNet(action_space_size(cfg.board_size)), cfg)


def test_gumbel_policy_is_valid_distribution():
    cfg = get_config("fast")
    cfg.mcts_sims = 16
    cfg.gumbel_n_considered = 8
    state = QuoridorState(cfg.board_size, cfg.walls_per_player).canonical()
    pi, v = _mcts(cfg).run_gumbel(state)
    mask = state.legal_actions_mask()
    assert pi.shape == mask.shape
    assert abs(float(pi.sum()) - 1.0) < 1e-5      # proper distribution
    assert np.all(pi[mask == 0] == 0.0)           # no mass on illegal actions
    assert -1.0 <= v <= 1.0
    print("ok: gumbel policy valid")


def test_gumbel_prefers_winning_move():
    cfg = get_config("fast")
    cfg.mcts_sims = 32
    cfg.gumbel_n_considered = 128             # consider every legal action
    s = QuoridorState(cfg.board_size, cfg.walls_per_player)
    s.pawns = [(cfg.board_size - 2, 2), (0, 2)]   # p0 one step from its goal row
    s.current_player = 0
    pi, v = _mcts(cfg).run_gumbel(s.canonical())
    win = OFFSET_TO_INDEX[(1, 0)]            # north step onto the goal row
    assert pi[win] == pi.max()              # search concentrates on the win
    assert v > 0                            # winning position has positive value
    print("ok: gumbel prefers winning move")


if __name__ == "__main__":
    test_gumbel_policy_is_valid_distribution()
    test_gumbel_prefers_winning_move()
    print("\nall mcts tests passed")
