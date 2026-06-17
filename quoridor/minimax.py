"""A no-training alpha-beta opponent using a shortest-path heuristic.

Branching in Quoridor is large (mostly wall placements), so candidate walls are
restricted to the most disruptive ones to keep search shallow but reasonable.
This gives an opponent that already plays a sensible game and is handy both as
a UI opponent and as a yardstick for the trained network.
"""

import math

from .game import (decode_action, h_wall_action, v_wall_action)

_INF = math.inf


def heuristic(state, root_player):
    """Positive favours ``root_player``. Path difference + walls in hand."""
    my = state.shortest_path_len(root_player)
    opp = state.shortest_path_len(1 - root_player)
    if my is None:
        return -1000.0
    if opp is None:
        return 1000.0
    score = (opp - my)
    score += 0.1 * (state.walls_left[root_player] - state.walls_left[1 - root_player])
    return float(score)


def _candidate_actions(state, max_walls):
    """Pawn moves plus the walls that most increase the opponent's path."""
    actions = list(state.legal_pawn_actions())
    if state.walls_left[state.current_player] <= 0 or max_walls <= 0:
        return actions

    opp = 1 - state.current_player
    base = state.shortest_path_len(opp) or 0
    scored = []
    w = state.N - 1
    for r in range(w):
        for c in range(w):
            for kind, act in (("h", h_wall_action(r, c, state.N)),
                              ("v", v_wall_action(r, c, state.N))):
                if not state.is_legal_wall(kind, r, c):
                    continue
                nxt = state.apply_action(act)
                gain = (nxt.shortest_path_len(opp) or 0) - base
                scored.append((gain, act))
    scored.sort(key=lambda t: t[0], reverse=True)
    actions.extend(act for _, act in scored[:max_walls])
    return actions


def search(state, depth, root_player, alpha, beta, max_walls):
    winner = state.winner()
    if winner is not None:
        return (1000.0 if winner == root_player else -1000.0), None
    if depth == 0:
        return heuristic(state, root_player), None

    maximizing = state.current_player == root_player
    best_action = None
    candidates = _candidate_actions(state, max_walls)

    if maximizing:
        value = -_INF
        for a in candidates:
            child = state.apply_action(a)
            score, _ = search(child, depth - 1, root_player, alpha, beta, max_walls)
            if score > value:
                value, best_action = score, a
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_action

    value = _INF
    for a in candidates:
        child = state.apply_action(a)
        score, _ = search(child, depth - 1, root_player, alpha, beta, max_walls)
        if score < value:
            value, best_action = score, a
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value, best_action


def best_action(state, depth=2, max_walls=12):
    _, action = search(state, depth, state.current_player, -_INF, _INF, max_walls)
    if action is None:  # fallback (e.g. depth 0): just pick something legal
        action = state.legal_actions()[0]
    return action
