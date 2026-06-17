"""Pygame UI to play Quoridor against the trained model or the minimax bot.

Examples
--------
    # play vs the no-training minimax bot (works without any checkpoint)
    python -m ui.play --opponent minimax --depth 2

    # play vs your trained AlphaZero net
    python -m ui.play --opponent model --model checkpoints/best.pt --sims 200

Controls
--------
    M            : move mode (default)
    H / V        : place a horizontal / vertical wall
    click panel  : tap the Move / Wall buttons to switch mode directly
    right-click  : flip the wall orientation (and enter wall mode)
    scroll wheel : rotate the wall orientation
    left-click   : move to a highlighted cell, or drop the previewed wall
    C            : cheat -- a deeper minimax suggests your best move
    A            : toggle auto-play -- keeps playing your best move each turn
    U            : undo (steps back to your previous turn)
    N            : new game
    ESC / Q      : quit
"""

import argparse
import math
import sys
import threading

import pygame

from quoridor import minimax
from quoridor.agents import MinimaxAgent, NeuralMCTSAgent, RandomAgent
from quoridor.config import Config
from quoridor.game import (QuoridorState, decode_action, h_wall_action,
                           v_wall_action, OFFSET_TO_INDEX, PAWN_OFFSETS)

# ---- layout ----
CELL = 56
GAP = 12
MARGIN = 40
PANEL = 240

BG = (24, 26, 32)
GRID = (54, 58, 70)
CELL_COLOR = (40, 44, 54)
HILITE = (70, 110, 90)
WALL_COLOR = (224, 188, 96)
WALL_OK = (110, 200, 130)
WALL_BAD = (210, 90, 90)
HINT = (190, 130, 255)
P0 = (90, 160, 250)
P1 = (240, 120, 120)
TEXT = (228, 230, 235)
MUTED = (150, 155, 165)


class QuoridorUI:
    def __init__(self, state, agent, human_player, cheat_depth=3, cheat_walls=None):
        self.state = state
        self.agent = agent
        self.human = human_player
        self.n = state.N
        self.mode = "move"          # "move" | "h" | "v"
        self.status = ""
        self.buttons = {}           # mode -> clickable Rect, filled while drawing
        self.undo_btn = None        # Rect for the Undo button, filled while drawing
        self.cheat_btn = None       # Rect for the Cheat button, filled while drawing
        self.auto_btn = None        # Rect for the Auto-move button
        self.history = []           # past states, newest last (for undo)

        # Cheat: a deliberately stronger minimax than the opponent.  Unlimited
        # walls means every legal wall slot is a search candidate.
        self.cheat_depth = cheat_depth
        self.cheat_walls = cheat_walls if cheat_walls is not None else 2 * (self.n - 1) ** 2
        self.hint = None            # suggested action for ``hint_for``
        self.hint_for = None        # the state the suggestion applies to
        self._hint_thread = None
        self._hint_box = {}         # background-thread result handoff
        self._hint_auto = False     # play the result instead of showing it
        self.auto_mode = False      # stay on, replaying the best move each turn

        board_px = self.n * CELL + (self.n - 1) * GAP
        self.origin = (MARGIN, MARGIN)
        # Keep the window tall enough for the side panel on small boards too.
        inner_h = max(board_px, 540)
        self.size = (MARGIN * 2 + board_px + PANEL, MARGIN * 2 + inner_h)

        pygame.init()
        pygame.display.set_caption("Quoridor")
        self.screen = pygame.display.set_mode(self.size)
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 16)
        self.big = pygame.font.SysFont("menlo,consolas,monospace", 26, bold=True)
        self.clock = pygame.time.Clock()

    # ---- geometry ----
    def cell_rect(self, r, c):
        ox, oy = self.origin
        return pygame.Rect(ox + c * (CELL + GAP),
                           oy + (self.n - 1 - r) * (CELL + GAP), CELL, CELL)

    def h_wall_rect(self, r, c):
        ox, oy = self.origin
        x = ox + c * (CELL + GAP)
        y = oy + (self.n - 1 - r) * (CELL + GAP) - GAP
        return pygame.Rect(x, y, 2 * CELL + GAP, GAP)

    def v_wall_rect(self, r, c):
        ox, oy = self.origin
        x = ox + c * (CELL + GAP) + CELL
        y = oy + (self.n - 1 - (r + 1)) * (CELL + GAP)
        return pygame.Rect(x, y, GAP, 2 * CELL + GAP)

    # ---- helpers ----
    def human_targets(self):
        """Map of destination (r, c) -> action for the human's legal moves."""
        targets = {}
        r, c = self.state.pawns[self.state.current_player]
        for a in self.state.legal_pawn_actions():
            dr, dc = PAWN_OFFSETS[a]
            targets[(r + dr, c + dc)] = a
        return targets

    def nearest_wall_slot(self, mouse):
        """Nearest legal wall slot of the current orientation, or None."""
        best, best_d = None, 1e9
        w = self.n - 1
        for r in range(w):
            for c in range(w):
                if not self.state.is_legal_wall(self.mode, r, c):
                    continue
                rect = (self.h_wall_rect if self.mode == "h" else self.v_wall_rect)(r, c)
                d = (rect.centerx - mouse[0]) ** 2 + (rect.centery - mouse[1]) ** 2
                if d < best_d:
                    best, best_d = (r, c), d
        if best is not None and best_d <= (CELL * 1.2) ** 2:
            return best
        return None

    # ---- input ----
    def is_human_turn(self):
        return self.state.current_player == self.human and not self.state.is_terminal()

    def rotate_wall(self):
        """Flip the previewed wall's orientation (entering wall mode if needed)."""
        self.mode = {"move": "h", "h": "v", "v": "h"}[self.mode]

    def commit(self, action):
        """Apply an action, remembering the prior state so it can be undone."""
        self.history.append(self.state)
        self.state = self.state.apply_action(action)
        self.clear_hint()

    def undo(self):
        """Step back to the human's previous turn (undoing the AI reply too)."""
        while self.history:
            self.state = self.history.pop()
            if self.state.current_player == self.human and not self.state.is_terminal():
                break
        self.clear_hint()

    # ---- cheat / hint ----
    def clear_hint(self):
        self.hint = None
        self.hint_for = None

    def hint_pending(self):
        return self._hint_thread is not None and self._hint_thread.is_alive()

    def toggle_auto(self):
        """Flip the auto-play toggle. It stays selected until clicked again,
        replaying the human's best move on every turn (incl. after the AI)."""
        self.auto_mode = not self.auto_mode

    def request_hint(self, auto=False):
        """Kick off a deeper minimax (on a thread) for the human's best move.

        ``auto`` plays the move once found; otherwise it is shown as a hint.
        """
        if not self.is_human_turn() or self.hint_pending():
            return
        if not auto and self.hint is not None and self.hint_for is self.state:
            return                                  # already solved this position
        target = self.state
        depth, walls = self.cheat_depth, self.cheat_walls
        self._hint_auto = auto
        box = self._hint_box = {}

        def work():
            box["action"] = minimax.best_action(target, depth, walls)
            box["state"] = target

        self._hint_thread = threading.Thread(target=work, daemon=True)
        self._hint_thread.start()

    def poll_hint(self):
        """Apply or show a finished background search if it still matches."""
        if self.hint_pending() or not self._hint_box:
            return
        action = self._hint_box.get("action")
        if self._hint_box.get("state") is self.state and action is not None:
            if self._hint_auto and self.is_human_turn() and self.auto_mode:
                self.commit(action)                 # play it for the human
            else:
                self.hint, self.hint_for = action, self.state
        self._hint_box = {}
        self._hint_thread = None
        self._hint_auto = False

    def handle_left_click(self, mouse):
        if self.undo_btn is not None and self.undo_btn.collidepoint(mouse):
            self.undo()
            return
        if self.cheat_btn is not None and self.cheat_btn.collidepoint(mouse):
            self.request_hint()
            return
        if self.auto_btn is not None and self.auto_btn.collidepoint(mouse):
            self.toggle_auto()
            return
        # Mode buttons take priority and work on any turn, so the human can
        # pick move / horizontal / vertical directly instead of using keys.
        for mode, rect in self.buttons.items():
            if rect.collidepoint(mouse):
                self.mode = mode
                return
        if not self.is_human_turn():
            return
        if self.mode == "move":
            for (r, c), a in self.human_targets().items():
                if self.cell_rect(r, c).collidepoint(mouse):
                    self.commit(a)
                    return
        else:
            slot = self.nearest_wall_slot(mouse)
            if slot is not None:
                r, c = slot
                a = (h_wall_action if self.mode == "h" else v_wall_action)(r, c, self.n)
                self.commit(a)

    # ---- drawing ----
    def draw(self, mouse):
        self.screen.fill(BG)
        targets = self.human_targets() if (self.is_human_turn() and self.mode == "move") else {}

        for r in range(self.n):
            for c in range(self.n):
                rect = self.cell_rect(r, c)
                pygame.draw.rect(self.screen, CELL_COLOR, rect, border_radius=6)
                if (r, c) in targets:
                    pygame.draw.rect(self.screen, HILITE, rect, border_radius=6)
                    pygame.draw.circle(self.screen, (150, 220, 170), rect.center, 7)

        # placed walls
        for (r, c) in self.state.h_walls:
            pygame.draw.rect(self.screen, WALL_COLOR, self.h_wall_rect(r, c), border_radius=3)
        for (r, c) in self.state.v_walls:
            pygame.draw.rect(self.screen, WALL_COLOR, self.v_wall_rect(r, c), border_radius=3)

        # wall preview
        if self.is_human_turn() and self.mode in ("h", "v"):
            slot = self.nearest_wall_slot(mouse)
            if slot is not None:
                r, c = slot
                rect = (self.h_wall_rect if self.mode == "h" else self.v_wall_rect)(r, c)
                pygame.draw.rect(self.screen, WALL_OK, rect, border_radius=3)

        self.draw_hint()

        # pawns
        for p in (0, 1):
            r, c = self.state.pawns[p]
            pygame.draw.circle(self.screen, P0 if p == 0 else P1,
                               self.cell_rect(r, c).center, CELL // 2 - 8)

        self.draw_panel()
        pygame.display.flip()

    def draw_hint(self):
        """Highlight the cheat's suggested move/wall for the current position."""
        if self.hint is None or self.hint_for is not self.state:
            return
        pulse = 3 + int(2 * (1 + math.sin(pygame.time.get_ticks() / 200.0)))
        kind = decode_action(self.hint, self.n)
        if kind[0] == "pawn":
            dr, dc = kind[1]
            r, c = self.state.pawns[self.human]
            center = self.cell_rect(r + dr, c + dc).center
            pygame.draw.circle(self.screen, HINT, center, CELL // 2 - 6, pulse)
        else:
            knd, r, c = kind
            rect = (self.h_wall_rect if knd == "h" else self.v_wall_rect)(r, c)
            pygame.draw.rect(self.screen, HINT, rect, border_radius=3)
            pygame.draw.rect(self.screen, (245, 235, 255), rect, width=2, border_radius=3)

    def draw_button(self, rect, label, active, accent, enabled=True):
        if not enabled:
            pygame.draw.rect(self.screen, BG, rect, border_radius=6)
            pygame.draw.rect(self.screen, GRID, rect, width=2, border_radius=6)
            col = MUTED
        else:
            pygame.draw.rect(self.screen, accent if active else CELL_COLOR, rect,
                             border_radius=6)
            pygame.draw.rect(self.screen, accent if active else GRID, rect,
                             width=2, border_radius=6)
            col = BG if active else TEXT
        surf = self.font.render(label, True, col)
        self.screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                                rect.centery - surf.get_height() // 2))

    def draw_panel(self):
        ox, oy = self.origin
        x = ox + self.n * CELL + (self.n - 1) * GAP + 24
        y = oy
        panel_w = PANEL - 24

        def line(txt, color=TEXT, dy=24, font=None):
            nonlocal y
            self.screen.blit((font or self.font).render(txt, True, color), (x, y))
            y += dy

        turn = self.state.current_player
        line("QUORIDOR", TEXT, 38, self.big)
        line(f"You: player {self.human}", P0 if self.human == 0 else P1)
        line(f"AI : {self.agent.name}", MUTED, 34)

        line(f"Walls  P0: {self.state.walls_left[0]}", P0)
        line(f"Walls  P1: {self.state.walls_left[1]}", P1, 34)

        if self.state.is_terminal():
            w = self.state.winner()
            who = "You win!" if w == self.human else "AI wins!"
            line(who, WALL_OK if w == self.human else WALL_BAD, 34, self.big)
        elif self.is_human_turn():
            line("Your turn", TEXT, 28)
        else:
            line("AI thinking...", MUTED, 28)

        # Clickable mode toolbar: pick move / horizontal / vertical directly.
        line("Action", MUTED, 22)
        bh = 30
        specs = [("move", "Move pawn", P0 if self.human == 0 else P1),
                 ("h", "Wall  —  (horizontal)", WALL_COLOR),
                 ("v", "Wall  |  (vertical)", WALL_COLOR)]
        self.buttons = {}
        for mode, label, accent in specs:
            rect = pygame.Rect(x, y, panel_w, bh)
            self.draw_button(rect, label, self.mode == mode, accent)
            self.buttons[mode] = rect
            y += bh + 6
        y += 6

        # Undo button: greyed out when there is nothing to undo.
        self.undo_btn = pygame.Rect(x, y, panel_w, bh)
        can_undo = bool(self.history)
        self.draw_button(self.undo_btn, "Undo" if can_undo else "Undo (nothing yet)",
                         False, WALL_BAD, enabled=can_undo)
        y += bh + 6

        # Cheat button: deeper minimax suggests the human's best move.
        self.cheat_btn = pygame.Rect(x, y, panel_w, bh)
        showing = self.hint is not None and self.hint_for is self.state
        thinking = self.hint_pending()
        if thinking and not self._hint_auto:
            cheat_label = "Cheat: thinking..."
        elif showing:
            cheat_label = "Cheat: showing best"
        else:
            cheat_label = f"Cheat (minimax d={self.cheat_depth})"
        self.draw_button(self.cheat_btn, cheat_label, showing, HINT,
                         enabled=self.is_human_turn())
        y += bh + 6

        # Auto-play toggle: stays selected until clicked again, replaying the
        # best move every turn.  Always clickable so it can be stopped anytime.
        self.auto_btn = pygame.Rect(x, y, panel_w, bh)
        if self.auto_mode:
            auto_label = "Auto: thinking..." if (thinking and self._hint_auto) \
                else "Auto: ON (click to stop)"
        else:
            auto_label = "Auto-play (off)"
        self.draw_button(self.auto_btn, auto_label, self.auto_mode, HINT)
        y += bh + 14

        for t in ["[M] move   [H]/[V] wall",
                  "right-click / scroll: flip wall",
                  "[C] hint  [A] auto  [U] undo",
                  "[N] new game   [Esc] quit"]:
            line(t, MUTED, 22)
        if self.status:
            y += 12
            line(self.status, WALL_BAD)

    # ---- loop ----
    def run(self):
        running = True
        while running:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_m:
                        self.mode = "move"
                    elif event.key == pygame.K_h:
                        self.mode = "h"
                    elif event.key == pygame.K_v:
                        self.mode = "v"
                    elif event.key == pygame.K_c:
                        self.request_hint()
                    elif event.key == pygame.K_a:
                        self.toggle_auto()
                    elif event.key == pygame.K_u:
                        self.undo()
                    elif event.key == pygame.K_n:
                        self.state = QuoridorState(self.n, self.state.walls_per_player)
                        self.history = []
                        self.clear_hint()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_left_click(mouse)
                    elif event.button == 3:        # right-click flips the wall
                        self.rotate_wall()
                elif event.type == pygame.MOUSEWHEEL:
                    self.rotate_wall()

            self.poll_hint()
            if self.auto_mode:
                self.request_hint(auto=True)        # keeps playing each turn
            self.draw(mouse)

            # AI plays after the frame is shown
            if not self.state.is_terminal() and self.state.current_player != self.human:
                pygame.time.wait(150)
                action = self.agent.select_action(self.state)
                self.commit(action)

            self.clock.tick(30)
        pygame.quit()


def build_agent(args):
    """Return (agent, board_size, walls_per_player)."""
    if args.opponent == "minimax":
        return MinimaxAgent(depth=args.depth), args.board_size, args.walls
    if args.opponent == "random":
        return RandomAgent(), args.board_size, args.walls

    # model: load checkpoint and match its board configuration
    import torch
    from quoridor.network import NeuralNet
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])
    if args.device:
        cfg.device = args.device
    if args.sims:
        cfg.mcts_sims = args.sims
    net = NeuralNet(cfg)
    net.load(args.model, load_optimizer=False)
    return NeuralMCTSAgent(net, cfg, temperature=0.0), cfg.board_size, cfg.walls_per_player


def main():
    parser = argparse.ArgumentParser(description="Play Quoridor vs an AI")
    parser.add_argument("--opponent", default="minimax",
                        choices=["model", "minimax", "random"])
    parser.add_argument("--model", default="checkpoints/best.pt")
    parser.add_argument("--sims", type=int, default=200, help="MCTS sims (model)")
    parser.add_argument("--depth", type=int, default=2, help="search depth (minimax)")
    parser.add_argument("--human", type=int, default=0, choices=[0, 1],
                        help="which player you control")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--walls", type=int, default=10)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "mps"])
    parser.add_argument("--cheat-depth", type=int, default=None,
                        help="minimax depth for the Cheat hint "
                             "(default: stronger than a minimax opponent)")
    parser.add_argument("--cheat-walls", type=int, default=None,
                        help="wall candidates the Cheat search considers "
                             "(default: unlimited)")
    args = parser.parse_args()

    try:
        agent, n, walls = build_agent(args)
    except FileNotFoundError:
        print(f"checkpoint not found: {args.model}\n"
              f"train one first (python -m quoridor.train) or use "
              f"--opponent minimax", file=sys.stderr)
        sys.exit(1)

    # Make the Cheat clearly stronger than the opponent: out-search a minimax
    # opponent by a ply, and default to a solid depth against anything else.
    cheat_depth = args.cheat_depth
    if cheat_depth is None:
        cheat_depth = max(3, args.depth + 1) if args.opponent == "minimax" else 3

    state = QuoridorState(n, walls)
    QuoridorUI(state, agent, args.human,
               cheat_depth=cheat_depth, cheat_walls=args.cheat_walls).run()


if __name__ == "__main__":
    main()
