"""AlphaZero training loop: self-play -> train -> gate -> repeat.

Run with:  python -m quoridor.train --preset fast
"""

import argparse
import os
import time

from .agents import MinimaxAgent, NeuralMCTSAgent
from .arena import arena
from .config import get_config
from .network import NeuralNet
from .replay_buffer import ReplayBuffer
from .selfplay import play_game


def _gate(candidate, best, config):
    """Return True if the freshly trained net should replace the best one."""
    cand_agent = NeuralMCTSAgent(candidate, config, temperature=0.0)
    best_agent = NeuralMCTSAgent(best, config, temperature=0.0)
    wins, losses, draws = arena(cand_agent, best_agent, config, config.eval_games)
    decisive = wins + losses
    rate = wins / decisive if decisive else 0.0
    print(f"  [gate] candidate vs best: {wins}-{losses}-{draws} "
          f"(win rate {rate:.2f}, need >= {config.eval_win_rate:.2f})")
    return rate >= config.eval_win_rate


def _eval_vs_minimax(net, config, depth=2):
    agent = NeuralMCTSAgent(net, config, temperature=0.0)
    mm = MinimaxAgent(depth=depth)
    wins, losses, draws = arena(agent, mm, config, config.eval_games)
    print(f"  [eval] net vs minimax(d={depth}): {wins}-{losses}-{draws}")


def train(config, resume=None, eval_minimax=True):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best = NeuralNet(config)
    print(f"device: {best.device} | board {config.board_size}x{config.board_size} "
          f"| action space {config.action_size}")

    start_iter = 0
    if resume and os.path.exists(resume):
        best.load(resume)
        print(f"resumed weights from {resume}")

    best.save(os.path.join(config.checkpoint_dir, "best.pt"))
    buffer = ReplayBuffer(config.replay_size)

    for it in range(start_iter, config.iterations):
        t0 = time.time()
        print(f"\n=== iteration {it + 1}/{config.iterations} ===")

        # 1) self-play with the current best net
        results = {0: 0, 1: 0, None: 0}
        for g in range(config.games_per_iter):
            samples, winner = play_game(best, config)
            buffer.add_many(samples)
            results[winner] += 1
        print(f"  self-play: p0={results[0]} p1={results[1]} draw={results[None]} "
              f"| buffer={len(buffer)} | {time.time() - t0:.0f}s")

        if len(buffer) < config.batch_size:
            print("  buffer too small, skipping training")
            continue

        # 2) train a candidate (a copy of best) on the buffer
        candidate = best.clone()
        losses = []
        for step in range(config.train_steps_per_iter):
            planes, pi, z = buffer.sample(config.batch_size)
            losses.append(candidate.train_step(planes, pi, z))
        if losses:
            mean = [sum(c) / len(c) for c in zip(*losses)]
            print(f"  train: loss={mean[0]:.3f} (policy={mean[1]:.3f} "
                  f"value={mean[2]:.3f}) over {len(losses)} steps")

        # 3) gate: keep the candidate only if it beats the current best
        candidate.save(os.path.join(config.checkpoint_dir, "latest.pt"))
        if _gate(candidate, best, config):
            best = candidate
            best.save(os.path.join(config.checkpoint_dir, "best.pt"))
            print("  -> promoted candidate to best")
        else:
            print("  -> kept previous best")

        if eval_minimax and (it + 1) % 5 == 0:
            _eval_vs_minimax(best, config)

    best.save(os.path.join(config.checkpoint_dir, "best.pt"))
    print("\ntraining complete -> checkpoints/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero Quoridor")
    parser.add_argument("--preset", default="fast", choices=["fast", "standard"])
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None, help="games per iteration")
    parser.add_argument("--sims", type=int, default=None, help="MCTS simulations")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "mps"])
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--no-eval-minimax", action="store_true")
    args = parser.parse_args()

    config = get_config(args.preset)
    if args.iterations is not None:
        config.iterations = args.iterations
    if args.games is not None:
        config.games_per_iter = args.games
    if args.sims is not None:
        config.mcts_sims = args.sims
    if args.device is not None:
        config.device = args.device

    train(config, resume=args.resume, eval_minimax=not args.no_eval_minimax)


if __name__ == "__main__":
    main()
