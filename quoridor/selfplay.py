"""Generate self-play training data with MCTS."""

import numpy as np

from .encoding import encode_state
from .game import QuoridorState, flip_action
from .mcts import MCTS


def play_game(net, config, mcts=None):
    """Play one self-play game.

    Returns (samples, winner) where each sample is
    (planes, policy_target, value_target) and value_target is filled in once
    the game ends, from the perspective of the player who was to move.
    """
    mcts = mcts or MCTS(net, config)
    state = QuoridorState(config.board_size, config.walls_per_player)
    history = []  # (planes, pi, player_to_move)

    move = 0
    while not state.is_terminal() and move < config.max_moves:
        canonical = state.canonical()
        counts = mcts.run_batched(canonical, add_noise=True)
        total = counts.sum()
        if total <= 0:  # no visits recorded; fall back to uniform legal
            counts = canonical.legal_actions_mask()
            total = counts.sum()
        pi = counts / total

        if move < config.temp_threshold:
            a_can = int(np.random.choice(len(pi), p=pi))
        else:
            a_can = int(np.argmax(counts))

        history.append((encode_state(canonical), pi, state.current_player))
        a_abs = a_can if state.current_player == 0 else flip_action(a_can, state.N)
        state = state.apply_action(a_abs)
        move += 1

    winner = state.winner()
    samples = []
    for planes, pi, player in history:
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        samples.append((planes, pi, np.float32(z)))
    return samples, winner
