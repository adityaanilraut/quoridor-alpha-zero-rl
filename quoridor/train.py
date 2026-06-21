"""AlphaZero training loop: self-play -> train -> gate -> repeat.

Run with:  python -m quoridor.train --preset fast
"""

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .agents import MinimaxAgent, NeuralMCTSAgent
from .arena import arena, arena_parallel
from .config import get_config
from .network import NeuralNet
from .replay_buffer import ReplayBuffer
from .selfplay import play_game


def _selfplay_worker(config_dict, checkpoint_path, worker_id):
    """Run a chunk of self-play games in a subprocess.

    Each worker loads its own copy of the network so there is no GIL / GPU
    contention between workers.  Returns the list of (planes, pi, z) samples.
    """
    config = Config.from_dict(config_dict)
    net = NeuralNet(config)
    net.load(checkpoint_path, load_optimizer=False)
    all_samples = []
    for _ in range(config.games_per_iter):
        samples, _ = play_game(net, config)
        all_samples.extend(samples)
    return all_samples


def _gate(config, candidate_path, best_path):
    """Return True if the freshly trained net should replace the best one."""
    wins, losses, draws = arena_parallel(candidate_path, best_path, config, config.eval_games)
    decisive = wins + losses
    rate = wins / decisive if decisive else 0.0
    print(f"  [gate] candidate vs best: {wins}-{losses}-{draws} "
          f"(win rate {rate:.2f}, need >= {config.eval_win_rate:.2f})")
    return rate >= config.eval_win_rate


def _gate_worker(config_dict, candidate_path, best_path):
    """Run gating in a subprocess to free the main process."""
    config = Config.from_dict(config_dict)
    return _gate(config, candidate_path, best_path)


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
        ckpt = best.load(resume)
        start_iter = int(ckpt.get("iteration", 0))
        print(f"resumed weights from {resume} (continuing at iteration "
              f"{start_iter + 1})")

    best_path = os.path.join(config.checkpoint_dir, "best.pt")
    best.save(best_path)
    buffer = ReplayBuffer(config.replay_size, recent_ratio=config.replay_recent_ratio)
    candidate = best.clone()  # persisted across iterations so training accumulates

    # Determine number of parallel workers
    num_workers = min(mp.cpu_count(), config.games_per_iter)
    print(f"using {num_workers} parallel workers for self-play "
          f"({config.games_per_iter} games across {num_workers} workers)")

    config_dict = config.to_dict()

    for it in range(start_iter, config.iterations):
        t0 = time.time()
        print(f"\n=== iteration {it + 1}/{config.iterations} ===")

        # 1) Parallel self-play
        best.save(best_path)  # ensure checkpoint is fresh for workers

        # Distribute games across workers
        base_games = config.games_per_iter // num_workers
        extra = config.games_per_iter % num_workers
        worker_games = [base_games + (1 if i < extra else 0)
                        for i in range(num_workers)]

        # Temporarily patch games_per_iter for each worker's chunk
        worker_configs = []
        for wg in worker_games:
            d = dict(config_dict)
            d["games_per_iter"] = wg
            worker_configs.append(d)

        all_samples = []
        with ProcessPoolExecutor(max_workers=num_workers,
                                 mp_context=mp.get_context("spawn")) as executor:
            futures = [
                executor.submit(_selfplay_worker, wc, best_path, i)
                for i, wc in enumerate(worker_configs)
            ]
            for f in as_completed(futures):
                samples = f.result()
                all_samples.extend(samples)

        buffer.add_many(all_samples)
        print(f"  self-play: {len(all_samples)} samples from "
              f"{config.games_per_iter} games "
              f"({time.time() - t0:.1f}s)")

        # 2) Train the candidate network (continues from previous iteration)
        losses = []
        for _ in range(config.train_steps_per_iter):
            if len(buffer) < config.batch_size:
                break
            planes, pi, z = buffer.sample(config.batch_size)
            losses.append(candidate.train_step(planes, pi, z))
        if losses:
            mean = [sum(c) / len(c) for c in zip(*losses)]
            print(f"  train: loss={mean[0]:.3f} (policy={mean[1]:.3f} "
                  f"value={mean[2]:.3f}) over {len(losses)} steps")

        # 3) Gate: keep the candidate only if it beats the current best
        candidate_path = os.path.join(config.checkpoint_dir, "latest.pt")
        candidate.save(candidate_path)

        # Run gating in a subprocess to free the main process
        promoted = _gate_worker(config_dict, candidate_path, best_path)
        if promoted:
            best = candidate.clone()
            best.save(best_path)
            print("  -> promoted candidate to best")
        else:
            # Revert candidate to best so next iteration starts from best weights
            candidate = best.clone()
            print("  -> kept previous best")

        if eval_minimax and (it + 1) % 5 == 0:
            _eval_vs_minimax(best, config)

        # Record progress so --resume can pick up at the next iteration.
        best.save(best_path, extra={"iteration": it + 1})

    best.save(best_path, extra={"iteration": config.iterations})
    print("\ntraining complete -> checkpoints/best.pt")


class Config:
    """Minimal shim so _selfplay_worker can reconstruct a config from a dict."""
    @classmethod
    def from_dict(cls, d):
        from .config import Config as Cfg
        return Cfg.from_dict(d)


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
