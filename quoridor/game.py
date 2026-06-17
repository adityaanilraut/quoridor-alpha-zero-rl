"""Quoridor game engine.

Coordinates use (row, col) with row 0 at player 0's home edge.

  * Player 0 starts at (0, N//2) and must reach row N-1.
  * Player 1 starts at (N-1, N//2) and must reach row 0.

Walls live on an (N-1) x (N-1) grid of slots:

  * A horizontal wall at (r, c) sits between rows r and r+1, spanning
    columns c and c+1. It blocks the two vertical edges below it.
  * A vertical wall at (r, c) sits between columns c and c+1, spanning
    rows r and r+1. It blocks the two horizontal edges beside it.

The board is parametrized by ``board_size`` so the exact same code can be
trained quickly on a small board (e.g. 5x5) before scaling to the real 9x9.
"""

from collections import deque

import numpy as np

# Pawn move offsets, indexed 0..11.  (dr, dc) in (row, col).
#   0-3  : single orthogonal steps   (N, S, E, W)
#   4-7  : straight jumps over a pawn (NN, SS, EE, WW)
#   8-11 : diagonal jumps            (NE, NW, SE, SW)
PAWN_OFFSETS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (2, 0), (-2, 0), (0, 2), (0, -2),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]
OFFSET_TO_INDEX = {off: i for i, off in enumerate(PAWN_OFFSETS)}
NUM_PAWN_ACTIONS = len(PAWN_OFFSETS)

# Action index of the vertically-flipped pawn move (used to convert actions
# between a player's canonical "moving north" frame and absolute coordinates).
#   N<->S, NN<->SS, NE<->SE, NW<->SW ; E/W/EE/WW unchanged.
FLIP_PAWN = [1, 0, 2, 3, 5, 4, 6, 7, 10, 11, 8, 9]


def action_space_size(n):
    """Total number of actions for an n x n board."""
    w = n - 1
    return NUM_PAWN_ACTIONS + 2 * w * w


def h_wall_action(r, c, n):
    return NUM_PAWN_ACTIONS + r * (n - 1) + c


def v_wall_action(r, c, n):
    w = n - 1
    return NUM_PAWN_ACTIONS + w * w + r * w + c


def decode_action(a, n):
    """Return ('pawn', (dr, dc)) or ('h'|'v', r, c)."""
    if a < NUM_PAWN_ACTIONS:
        return ("pawn", PAWN_OFFSETS[a])
    w = n - 1
    a -= NUM_PAWN_ACTIONS
    if a < w * w:
        return ("h", a // w, a % w)
    a -= w * w
    return ("v", a // w, a % w)


def flip_action(a, n):
    """Map an action index under a vertical board flip (its own inverse)."""
    if a < NUM_PAWN_ACTIONS:
        return FLIP_PAWN[a]
    kind, r, c = decode_action(a, n)
    if kind == "h":
        return h_wall_action(n - 2 - r, c, n)
    return v_wall_action(n - 2 - r, c, n)


class QuoridorState:
    """An immutable-by-convention Quoridor position (mutated only via clone)."""

    __slots__ = ("N", "walls_per_player", "pawns", "walls_left",
                 "h_walls", "v_walls", "current_player", "move_count")

    def __init__(self, board_size=9, walls_per_player=10):
        self.N = board_size
        self.walls_per_player = walls_per_player
        mid = board_size // 2
        self.pawns = [(0, mid), (board_size - 1, mid)]
        self.walls_left = [walls_per_player, walls_per_player]
        self.h_walls = set()
        self.v_walls = set()
        self.current_player = 0
        self.move_count = 0

    # ------------------------------------------------------------------ #
    # Basic helpers
    # ------------------------------------------------------------------ #
    def clone(self):
        s = QuoridorState(self.N, self.walls_per_player)
        s.pawns = list(self.pawns)
        s.walls_left = list(self.walls_left)
        s.h_walls = set(self.h_walls)
        s.v_walls = set(self.v_walls)
        s.current_player = self.current_player
        s.move_count = self.move_count
        return s

    def _in_bounds(self, r, c):
        return 0 <= r < self.N and 0 <= c < self.N

    def _passage_open(self, r, c, dr, dc):
        """True if a single step (dr, dc) from (r, c) is not blocked by a wall.

        Ignores board bounds and pawns; the caller checks those.
        """
        h, v = self.h_walls, self.v_walls
        if dr == 1 and dc == 0:        # north (row + 1)
            return (r, c) not in h and (r, c - 1) not in h
        if dr == -1 and dc == 0:       # south (row - 1)
            return (r - 1, c) not in h and (r - 1, c - 1) not in h
        if dr == 0 and dc == 1:        # east (col + 1)
            return (r, c) not in v and (r - 1, c) not in v
        if dr == 0 and dc == -1:       # west (col - 1)
            return (r, c - 1) not in v and (r - 1, c - 1) not in v
        raise ValueError(f"not a unit step: {(dr, dc)}")

    @staticmethod
    def _perp(dr, dc):
        return [(0, 1), (0, -1)] if dc == 0 else [(1, 0), (-1, 0)]

    def goal_row(self, player):
        return self.N - 1 if player == 0 else 0

    # ------------------------------------------------------------------ #
    # Legal moves
    # ------------------------------------------------------------------ #
    def legal_pawn_actions(self):
        cp = self.current_player
        r, c = self.pawns[cp]
        opp = self.pawns[1 - cp]
        actions = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not self._in_bounds(nr, nc) or not self._passage_open(r, c, dr, dc):
                continue
            if (nr, nc) != opp:
                actions.append(OFFSET_TO_INDEX[(dr, dc)])
                continue
            # Opponent is adjacent: try a straight jump, else diagonals.
            br, bc = r + 2 * dr, c + 2 * dc
            if self._in_bounds(br, bc) and self._passage_open(opp[0], opp[1], dr, dc):
                actions.append(OFFSET_TO_INDEX[(2 * dr, 2 * dc)])
            else:
                for pdr, pdc in self._perp(dr, dc):
                    tr, tc = opp[0] + pdr, opp[1] + pdc
                    if self._in_bounds(tr, tc) and self._passage_open(opp[0], opp[1], pdr, pdc):
                        actions.append(OFFSET_TO_INDEX[(dr + pdr, dc + pdc)])
        return actions

    def _wall_conflicts(self, kind, r, c):
        h, v = self.h_walls, self.v_walls
        if kind == "h":
            return ((r, c) in h or (r, c - 1) in h or (r, c + 1) in h
                    or (r, c) in v)
        return ((r, c) in v or (r - 1, c) in v or (r + 1, c) in v
                or (r, c) in h)

    def _has_path(self, player):
        """BFS: can ``player`` still reach its goal row (ignoring pawns)?"""
        n = self.N
        goal = self.goal_row(player)
        start = self.pawns[player]
        seen = [[False] * n for _ in range(n)]
        dq = deque([start])
        seen[start[0]][start[1]] = True
        while dq:
            r, c = dq.popleft()
            if r == goal:
                return True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < n and 0 <= nc < n and not seen[nr][nc]
                        and self._passage_open(r, c, dr, dc)):
                    seen[nr][nc] = True
                    dq.append((nr, nc))
        return False

    def is_legal_wall(self, kind, r, c):
        cp = self.current_player
        if self.walls_left[cp] <= 0:
            return False
        w = self.N - 1
        if not (0 <= r < w and 0 <= c < w):
            return False
        if self._wall_conflicts(kind, r, c):
            return False
        bucket = self.h_walls if kind == "h" else self.v_walls
        bucket.add((r, c))
        ok = self._has_path(0) and self._has_path(1)
        bucket.discard((r, c))
        return ok

    def legal_actions_mask(self):
        mask = np.zeros(action_space_size(self.N), dtype=np.float32)
        for a in self.legal_pawn_actions():
            mask[a] = 1.0
        if self.walls_left[self.current_player] > 0:
            w = self.N - 1
            for r in range(w):
                for c in range(w):
                    if self.is_legal_wall("h", r, c):
                        mask[h_wall_action(r, c, self.N)] = 1.0
                    if self.is_legal_wall("v", r, c):
                        mask[v_wall_action(r, c, self.N)] = 1.0
        return mask

    def legal_actions(self):
        return [int(a) for a in np.nonzero(self.legal_actions_mask())[0]]

    # ------------------------------------------------------------------ #
    # Transitions / terminal
    # ------------------------------------------------------------------ #
    def apply_action(self, a):
        s = self.clone()
        cp = s.current_player
        kind = decode_action(a, s.N)
        if kind[0] == "pawn":
            dr, dc = kind[1]
            r, c = s.pawns[cp]
            s.pawns[cp] = (r + dr, c + dc)
        elif kind[0] == "h":
            s.h_walls.add((kind[1], kind[2]))
            s.walls_left[cp] -= 1
        else:
            s.v_walls.add((kind[1], kind[2]))
            s.walls_left[cp] -= 1
        s.current_player = 1 - cp
        s.move_count += 1
        return s

    def winner(self):
        if self.pawns[0][0] == self.N - 1:
            return 0
        if self.pawns[1][0] == 0:
            return 1
        return None

    def is_terminal(self):
        return self.winner() is not None

    def shortest_path_len(self, player):
        """Distance to the goal row respecting walls, or None if unreachable."""
        n = self.N
        goal = self.goal_row(player)
        start = self.pawns[player]
        dist = [[-1] * n for _ in range(n)]
        dq = deque([start])
        dist[start[0]][start[1]] = 0
        while dq:
            r, c = dq.popleft()
            if r == goal:
                return dist[r][c]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < n and 0 <= nc < n and dist[nr][nc] < 0
                        and self._passage_open(r, c, dr, dc)):
                    dist[nr][nc] = dist[r][c] + 1
                    dq.append((nr, nc))
        return None

    # ------------------------------------------------------------------ #
    # Canonical (current-player) perspective
    # ------------------------------------------------------------------ #
    def canonical(self):
        """Return an equivalent state with player 0 to move.

        If it is already player 0's turn this is just a clone.  Otherwise the
        board is flipped vertically and the pawns swapped, so the player to
        move always starts near row 0 and advances toward row N-1.  This lets
        the network learn a single orientation.
        """
        if self.current_player == 0:
            return self.clone()
        n = self.N
        s = self.clone()
        p0, p1 = self.pawns
        s.pawns = [(n - 1 - p1[0], p1[1]), (n - 1 - p0[0], p0[1])]
        s.walls_left = [self.walls_left[1], self.walls_left[0]]
        s.h_walls = {(n - 2 - r, c) for (r, c) in self.h_walls}
        s.v_walls = {(n - 2 - r, c) for (r, c) in self.v_walls}
        s.current_player = 0
        return s

    def __repr__(self):
        return (f"QuoridorState(N={self.N}, p0={self.pawns[0]}, p1={self.pawns[1]}, "
                f"walls={self.walls_left}, turn={self.current_player})")
