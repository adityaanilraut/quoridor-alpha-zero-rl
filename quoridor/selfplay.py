"""Generate self-play training data with MCTS."""

import numpy as np

from .encoding import encode_state
from .game import QuoridorState, flip_action
from .mcts import MCTS


def play_game(net, config, mcts=None):
    """Play one self-play game.

    Returns ``(samples, winner)`` where each sample is
    ``(planes, canonical_state, policy_target, outcome, search_value)``:

      * ``canonical_state`` lets the position be re-searched later (reanalyze).
      * ``outcome`` is the eventual game result from the mover's perspective.
      * ``search_value`` is the MCTS root value (or the outcome when Gumbel
        search is disabled).  The replay buffer mixes the two into the value
        target via ``mixed_value_lambda``.
    """
    mcts = mcts or MCTS(net, config)
    use_gumbel = getattr(config, "use_gumbel", True)
    state = QuoridorState(config.board_size, config.walls_per_player)
    history = []  # (planes, canonical_state, pi, player_to_move, search_value)

    move = 0
    while not state.is_terminal() and move < config.max_moves:
        canonical = state.canonical()
        if use_gumbel:
            pi, v_search = mcts.run_gumbel(canonical)
        else:
            counts = mcts.run_batched(canonical, add_noise=True)
            total = counts.sum()
            if total <= 0:  # no visits recorded; fall back to uniform legal
                counts = canonical.legal_actions_mask()
                total = counts.sum()
            pi = counts / total
            v_search = None

        if pi.sum() <= 0:  # safety: no policy mass -> uniform over legal moves
            pi = canonical.legal_actions_mask()
            pi = pi / pi.sum()

        if move < config.temp_threshold:
            a_can = int(np.random.choice(len(pi), p=pi))
        else:
            a_can = int(np.argmax(pi))

        history.append((encode_state(canonical), canonical, pi,
                        state.current_player, v_search))
        a_abs = a_can if state.current_player == 0 else flip_action(a_can, state.N)
        state = state.apply_action(a_abs)
        move += 1

    winner = state.winner()
    samples = []
    for planes, canon, pi, player, v_search in history:
        outcome = 0.0 if winner is None else (1.0 if winner == player else -1.0)
        v_store = outcome if v_search is None else float(v_search)
        samples.append((planes, canon, pi.astype(np.float32),
                        np.float32(outcome), np.float32(v_store)))
    return samples, winner
