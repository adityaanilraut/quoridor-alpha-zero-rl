"""Hyperparameters and presets.

Two presets are provided:

  * ``fast``    : 5x5 board, tiny net, few sims.  Trains a competent bot in
    minutes on a laptop CPU -- use it to validate the whole pipeline.
  * ``standard``: full 9x9 board.  Stronger but much slower; expect to leave
    it running for hours.
"""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    name: str = "standard"

    # --- board ---
    board_size: int = 9
    walls_per_player: int = 10

    # --- network ---
    channels: int = 64
    res_blocks: int = 5

    # --- MCTS ---
    mcts_sims: int = 120
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # --- self-play ---
    games_per_iter: int = 30
    temp_threshold: int = 12      # moves before play becomes greedy
    max_moves: int = 200          # safety cap -> declared a draw

    # --- training ---
    iterations: int = 100
    replay_size: int = 60_000
    replay_recent_ratio: float = 0.5  # fraction of each batch drawn from newest data
    batch_size: int = 256
    train_steps_per_iter: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # --- evaluation / gating ---
    eval_games: int = 20
    eval_win_rate: float = 0.55   # promote new net if it scores >= this

    # --- io / device ---
    device: str = "auto"          # auto | cpu | mps
    checkpoint_dir: str = "checkpoints"

    @property
    def action_size(self):
        from .game import action_space_size
        return action_space_size(self.board_size)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


def fast_config():
    """Small 5x5 preset for quickly validating the full loop."""
    return Config(
        name="fast",
        board_size=5,
        walls_per_player=3,
        channels=32,
        res_blocks=3,
        mcts_sims=60,
        games_per_iter=30,
        temp_threshold=8,
        max_moves=80,
        iterations=60,
        replay_size=20_000,
        replay_recent_ratio=0.5,
        batch_size=128,
        train_steps_per_iter=1000,
        eval_games=20,
    )


def standard_config():
    return Config()


PRESETS = {"fast": fast_config, "standard": standard_config}


def get_config(name="standard"):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return PRESETS[name]()
