"""Board <-> tensor encoding for the neural network.

States are expected to be in canonical form (player 0 to move). The encoding
is always from the perspective of the player to move:

  plane 0 : current player's pawn (one-hot)
  plane 1 : opponent's pawn (one-hot)
  plane 2 : horizontal walls
  plane 3 : vertical walls
  plane 4 : current player's walls remaining (constant, normalized)
  plane 5 : opponent's walls remaining (constant, normalized)
"""

import numpy as np

NUM_PLANES = 6


def encode_state(state):
    n = state.N
    planes = np.zeros((NUM_PLANES, n, n), dtype=np.float32)
    cp = state.current_player
    pr, pc = state.pawns[cp]
    orr, occ = state.pawns[1 - cp]
    planes[0, pr, pc] = 1.0
    planes[1, orr, occ] = 1.0
    for r, c in state.h_walls:
        planes[2, r, c] = 1.0
    for r, c in state.v_walls:
        planes[3, r, c] = 1.0
    wpp = max(state.walls_per_player, 1)
    planes[4, :, :] = state.walls_left[cp] / wpp
    planes[5, :, :] = state.walls_left[1 - cp] / wpp
    return planes
