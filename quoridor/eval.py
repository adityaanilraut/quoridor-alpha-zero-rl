"""Evaluate a trained model against the minimax baseline.

Usage:
    python -m quoridor.eval --checkpoint checkpoints/best.pt
    python -m quoridor.eval --checkpoint checkpoints/best.pt --preset standard
    python -m quoridor.eval --checkpoint checkpoints/best.pt --games 50 --depth 3
"""

import argparse

from .agents import MinimaxAgent, NeuralMCTSAgent
from .arena import arena
from .config import get_config
from .network import NeuralNet


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model vs minimax"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="path to model checkpoint (default: checkpoints/best.pt)")
    parser.add_argument("--preset", default="fast", choices=["fast", "standard"],
                        help="config preset matching the checkpoint (default: fast)")
    parser.add_argument("--games", type=int, default=20,
                        help="number of games to play (default: 20)")
    parser.add_argument("--depth", type=int, default=2,
                        help="minimax search depth (default: 2)")
    parser.add_argument("--sims", type=int, default=None,
                        help="MCTS simulations (default: from config)")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "mps"])
    parser.add_argument("--verbose", action="store_true",
                        help="print per-game results")
    args = parser.parse_args()

    config = get_config(args.preset)
    if args.sims is not None:
        config.mcts_sims = args.sims
    if args.device is not None:
        config.device = args.device

    print(f"Loading checkpoint: {args.checkpoint}")
    net = NeuralNet(config)
    net.load(args.checkpoint, load_optimizer=False)
    print(f"  board: {config.board_size}x{config.board_size}  "
          f"sims: {config.mcts_sims}  device: {net.device}")

    model_agent = NeuralMCTSAgent(net, config, temperature=0.0)
    mm_agent = MinimaxAgent(depth=args.depth)

    print(f"Playing {args.games} games: model vs minimax(depth={args.depth}) ...")
    wins, losses, draws = arena(model_agent, mm_agent, config,
                                n_games=args.games, verbose=args.verbose)

    decisive = wins + losses
    rate = wins / decisive if decisive else 0.0
    print(f"\nResults ({args.games} games):")
    print(f"  Model wins:  {wins}")
    print(f"  Minimax wins: {losses}")
    print(f"  Draws:       {draws}")
    print(f"  Model win rate: {rate:.2%}")


if __name__ == "__main__":
    main()
