# Quoridor AlphaZero

A from-scratch [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404)-style
agent that learns to play **Quoridor** by self-play — plus a Pygame board so you
can play against it. Tuned to actually train on a laptop (Apple Silicon / 8 GB).

The same code runs on a small **5×5** board (trains a competent bot in ~an hour
on CPU) and on the full **9×9** board (much stronger, much slower).

---

## Why AlphaZero for Quoridor?

Quoridor is a two-player, perfect-information, deterministic, zero-sum game —
exactly the setting AlphaZero was designed for. No human games or hand-coded
strategy are needed: a single network learns both

* a **policy** — a probability over moves, and
* a **value** — who is winning,

and a Monte-Carlo Tree Search (MCTS) uses the network to look ahead. Games are
played by the agent against itself; the outcomes train the network; repeat. The
network discovers wall tactics and pathing on its own.

A no-training **alpha-beta minimax** bot (shortest-path heuristic) is also
included as an immediate opponent and as a strength yardstick.

---

## Project layout

```
quoridor/
  game.py          rules engine: moves, jumps, wall legality (path guarantee), win
  encoding.py      board -> tensor planes (canonical / side-to-move perspective)
  network.py       policy + value ResNet and a torch wrapper (predict / train / io)
  mcts.py          PUCT MCTS guided by the network
  selfplay.py      generate (state, policy, outcome) training samples
  replay_buffer.py fixed-size sample store
  arena.py         pit two agents to measure relative strength
  train.py         the self-play -> train -> gate loop  (entry point)
  minimax.py       no-training alpha-beta opponent
  agents.py        RandomAgent / MinimaxAgent / NeuralMCTSAgent
  config.py        hyperparameters + `fast` (5x5) and `standard` (9x9) presets
ui/play.py         Pygame board to play vs the model or minimax  (entry point)
tests/test_game.py engine sanity checks
checkpoints/       saved models (best.pt / latest.pt)
```

---

## Install

Requires Python 3.10+ (tested on 3.14).

```bash
cd /Users/adityaraut/Desktop/Quoridor
python3 -m pip install -r requirements.txt
```

`torch` ships with the Metal (MPS) backend on Apple Silicon out of the box.
`pygame-ce` is a maintained drop-in for `pygame` with current wheels (the import
name is still `pygame`).

Verify:

```bash
python3 -m tests.test_game        # engine tests should all pass
```

---

## Quick start — play right now (no training)

The minimax bot needs no checkpoint, so you can play immediately:

```bash
python3 -m ui.play --opponent minimax --depth 2
```

**Controls**

| Key | Action |
|-----|--------|
| `M` | move mode (default) — click a highlighted cell to move |
| `H` | horizontal-wall mode — hover a slot, click to place |
| `V` | vertical-wall mode |
| right-click / scroll | flip the previewed wall's orientation |
| `C` | cheat — a deeper minimax suggests (and highlights) your best move |
| `A` | auto move — the same search just plays your best move for you |
| `U` | undo — steps back to your previous turn |
| `N` | new game |
| `Esc` / `Q` | quit |

You can also click the **Move / Wall / Undo / Cheat / Auto move** buttons in the
side panel instead of using the keys.

The **Cheat** and **Auto move** features run a stronger minimax than the opponent
(deeper search and unlimited wall candidates), on a background thread so the
window stays responsive while it thinks. `C` highlights the best move; `A` plays
it. Tune the search with `--cheat-depth` and `--cheat-walls`.

You are the blue pawn (player 0, bottom) by default; pass `--human 1` to play red.

---

## Training

### 1. Validate the whole pipeline on 5×5 (do this first)

```bash
python3 -m quoridor.train --preset fast --device cpu
```

Each **iteration** does:

1. **Self-play** — play `games_per_iter` games with the current best net + MCTS,
   storing `(board, MCTS-policy, game-outcome)` samples in the replay buffer.
2. **Train** — a *candidate* (copy of best) does `train_steps_per_iter` SGD steps
   on random minibatches (policy cross-entropy + value MSE).
3. **Gate** — candidate plays an `eval_games` match vs the current best; it is
   promoted to `best.pt` only if it wins ≥ `eval_win_rate`. This keeps training
   monotonic and is the standard AlphaZero safeguard.

Checkpoints are written every iteration to `checkpoints/best.pt` (the strongest
so far) and `checkpoints/latest.pt` (most recent candidate).

### 2. Train on the full 9×9 board

```bash
python3 -m quoridor.train --preset standard --device cpu
```

### Useful flags

```
--preset {fast,standard}   board + hyperparameter preset (default: fast)
--iterations N             number of train iterations
--games N                  self-play games per iteration
--sims N                   MCTS simulations per move
--device {auto,cpu,mps}    compute device
--resume checkpoints/best.pt   continue from a saved model
--no-eval-minimax          skip periodic minimax benchmark
```

Example — a lighter 9×9 run:

```bash
python3 -m quoridor.train --preset standard --games 16 --sims 80 --device cpu
```

### What to expect on an M3 / 8 GB (measured, CPU)

| Board | MCTS sims | Self-play speed | Rough time / iteration\* |
|-------|-----------|-----------------|--------------------------|
| 5×5 (`fast`) | 60 | ~2 s / game | ~1.5–3 min |
| 9×9 (`standard`) | 120 | ~20 s / game (~190 ms/move) | ~15–25 min |

\*Includes self-play + training + the gating match. A *playable* 5×5 bot appears
within ~an hour; strong 9×9 play needs many iterations — run it overnight, or
lower `--games`/`--sims` to iterate faster. Training is resumable, so you can
stop and `--resume checkpoints/best.pt` later.

### Device choice (important on this machine)

MCTS does **many tiny single-board inferences**, where MPS's per-call overhead
often makes it *slower* than CPU for this small network — so **`--device cpu` is
recommended** for self-play-heavy training here. `--device mps` mainly helps the
batched training step; for a net this small the difference is minor. `auto`
prefers MPS if present.

### Speed knobs

* Start on 5×5 (`fast`) — same code, dramatically faster.
* Lower `--sims` (search cost is linear in sims) and `--games`.

---

## Evaluation: trained model vs minimax

After training (or with a pre-trained checkpoint), pit the model against the
alpha-beta minimax baseline:

```bash
python3 -m quoridor.eval --checkpoint checkpoints/best.pt --preset fast
```

### Flags

```
--checkpoint PATH   model checkpoint (default: checkpoints/best.pt)
--preset {fast,standard}   must match the one used during training
--games N           number of games to play (default: 20)
--depth N           minimax search depth (default: 2)
--sims N            MCTS simulations (default: from config)
--device {auto,cpu,mps}
--verbose           print per-game results
```

Example — 50 games against a deeper minimax on the full board:

```bash
python3 -m quoridor.eval --preset standard --games 50 --depth 3 --verbose
```
* The cost driver on 9×9 is wall-legality checking (a BFS per candidate wall);
  fewer sims is the most effective lever.

---

## Inference — play against your trained model

```bash
python3 -m ui.play --opponent model --model checkpoints/best.pt --sims 200
```

`--sims` controls how hard the model thinks per move (more = stronger + slower).
The UI reads the board size and wall count from the checkpoint, so a model
trained on 5×5 automatically opens a 5×5 board. Add `--device cpu` if MPS feels
laggy for single-move inference.

Programmatic inference:

```python
import torch
from quoridor.config import Config
from quoridor.network import NeuralNet
from quoridor.agents import NeuralMCTSAgent
from quoridor.game import QuoridorState

ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
cfg = Config.from_dict(ckpt["config"]); cfg.device = "cpu"; cfg.mcts_sims = 200
net = NeuralNet(cfg); net.load("checkpoints/best.pt", load_optimizer=False)
agent = NeuralMCTSAgent(net, cfg, temperature=0.0)   # greedy

state = QuoridorState(cfg.board_size, cfg.walls_per_player)
action = agent.select_action(state)          # action index in 0..action_size-1
state = state.apply_action(action)
```

---

## Evaluation

Measure strength by pitting agents against each other with alternating colors:

```python
from quoridor.config import get_config
from quoridor.network import NeuralNet
from quoridor.agents import NeuralMCTSAgent, MinimaxAgent
from quoridor.arena import arena

cfg = get_config("fast"); cfg.device = "cpu"
net = NeuralNet(cfg); net.load("checkpoints/best.pt", load_optimizer=False)

wins, losses, draws = arena(
    NeuralMCTSAgent(net, cfg), MinimaxAgent(depth=2), cfg, n_games=20)
print(f"net vs minimax: {wins}-{losses}-{draws}")
```

`train.py` also runs this minimax benchmark automatically every 5 iterations.

---

## How it works (internals)

**State & rules** (`game.py`). Coordinates are `(row, col)`; player 0 starts at
the bottom and aims for the top. Walls live on an `(N-1)×(N-1)` grid; a wall is
legal only if it doesn't overlap/cross an existing wall **and** both pawns still
have *some* path to their goal (verified by BFS) — the rule that makes Quoridor
interesting. Pawn moves include orthogonal steps, straight jumps over an adjacent
pawn, and the diagonal jumps allowed when the straight jump is blocked.

**Canonical form.** Every position is presented to the network from the side-to-
move's perspective: if it's player 1's turn the board is flipped vertically and
the pawns swapped, so the mover always advances "north". This halves what the
network must learn. Actions are translated back to absolute coordinates with
`flip_action` (a self-inverse mapping).

**Encoding** (`encoding.py`). 6 planes of `N×N`: your pawn, opponent pawn,
horizontal walls, vertical walls, and two constant planes for each player's walls
remaining.

**Action space.** `12 + 2·(N-1)²` actions: 12 pawn moves (4 steps + 4 jumps +
4 diagonals) and a slot for every horizontal/vertical wall. (9×9 → 140; 5×5 → 44.)

**Network** (`network.py`). A conv stem + residual blocks feeding a policy head
(logits over all actions) and a value head (`tanh`, who's winning). Small by
design: `channels=64, blocks=5` for 9×9.

**MCTS** (`mcts.py`). PUCT selection (`Q + c_puct·P·√ΣN/(1+N)`), the network
supplies priors `P` and leaf value `v`, no random rollouts. Dirichlet noise at
the root during self-play for exploration. The move-selection target is the
visit-count distribution.

**Training** (`train.py`). Loss = policy cross-entropy against the MCTS visit
distribution + MSE between the value head and the eventual game result (from each
position's side-to-move perspective). The gating match prevents regressions.

---

## Tuning (`quoridor/config.py`)

| Field | Meaning | `fast` | `standard` |
|-------|---------|-------|-----------|
| `board_size` / `walls_per_player` | board + walls each | 5 / 3 | 9 / 10 |
| `channels` / `res_blocks` | network size | 32 / 3 | 64 / 5 |
| `mcts_sims` | search per move | 60 | 120 |
| `c_puct` | exploration in MCTS | 1.5 | 1.5 |
| `dirichlet_alpha` / `_eps` | root exploration noise | 0.3 / 0.25 | 0.3 / 0.25 |
| `games_per_iter` | self-play games / iter | 20 | 30 |
| `temp_threshold` | exploratory plies before greedy | 8 | 12 |
| `train_steps_per_iter` / `batch_size` | SGD per iter | 200 / 128 | 400 / 256 |
| `eval_games` / `eval_win_rate` | gating match | 20 / 0.55 | 20 / 0.55 |
| `replay_size` | sample buffer | 20k | 60k |

---

## Tips & troubleshooting

* **Bot wanders / games hit the move cap early in training.** Expected — an
  untrained net has no idea where the goal is. It sharpens after a few promoted
  iterations. Watch the `[gate]` win rates and the periodic minimax benchmark.
* **9×9 feels too slow.** Lower `--sims` first, then `--games`; or develop on
  `--preset fast` and only move to 9×9 once you're happy with the pipeline.
* **Pygame window won't open over SSH/headless.** It needs a display; run locally.
* **`pygame` install issues on Python 3.14.** Use `pygame-ce` (in requirements).
* **MPS slower than CPU.** Normal for tiny single inferences — use `--device cpu`.

---

## Roadmap ideas

* Batch MCTS leaf evaluations to better use MPS.
* Move the hot path (legal-move / wall-BFS generation) to C/Numba for faster 9×9.
* Multiprocess self-play workers.
* Optional opening randomization and resignation to speed up self-play.
