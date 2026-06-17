"""AlphaZero-style Quoridor engine, training pipeline and agents."""

from .config import Config, get_config
from .game import QuoridorState

__all__ = ["Config", "get_config", "QuoridorState"]
