"""Pit two agents against each other to measure relative strength."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from .game import QuoridorState


def play_match(agent0, agent1, config, max_moves=None):
    """Play a single game; agent0 is player 0. Returns the winner (0/1/None)."""
    max_moves = max_moves or config.max_moves
    state = QuoridorState(config.board_size, config.walls_per_player)
    agents = (agent0, agent1)
    move = 0
    while not state.is_terminal() and move < max_moves:
        action = agents[state.current_player].select_action(state)
        state = state.apply_action(action)
        move += 1
    return state.winner()


def _result_code(winner, swap):
    """Map a board winner (player 0/1/None) to a result code.

    ``0`` => agent_a won, ``1`` => agent_b won, ``None`` => draw.  When
    ``swap`` is True, agent_a played as player 1 and agent_b as player 0, so
    the player index has to be inverted to recover who the result belongs to.
    """
    if winner is None:
        return None
    if swap:
        # agent_a is player 1, agent_b is player 0
        return 0 if winner == 1 else 1
    # agent_a is player 0, agent_b is player 1
    return 0 if winner == 0 else 1


def _match_worker(args):
    """Pickleable wrapper for parallel arena games.

    Receives checkpoint paths (not live agent objects) so MPS tensors are
    never serialized across processes.  Each worker loads its own copy.
    """
    config_dict, path_a, path_b, swap = args
    from .agents import NeuralMCTSAgent
    from .config import Config
    from .network import NeuralNet

    config = Config.from_dict(config_dict)
    net_a = NeuralNet(config)
    net_b = NeuralNet(config)
    net_a.load(path_a, load_optimizer=False)
    net_b.load(path_b, load_optimizer=False)
    agent_a = NeuralMCTSAgent(net_a, config, temperature=0.0)
    agent_b = NeuralMCTSAgent(net_b, config, temperature=0.0)

    if swap:
        winner = play_match(agent_b, agent_a, config)
    else:
        winner = play_match(agent_a, agent_b, config)
    return _result_code(winner, swap)


def arena(agent_a, agent_b, config, n_games=20, verbose=False):
    """Play n_games with alternating colors. Returns (wins_a, wins_b, draws)."""
    wins_a = wins_b = draws = 0
    for g in range(n_games):
        if g % 2 == 0:
            winner = play_match(agent_a, agent_b, config)
            a_won = winner == 0
        else:
            winner = play_match(agent_b, agent_a, config)
            a_won = winner == 1
        if winner is None:
            draws += 1
        elif a_won:
            wins_a += 1
        else:
            wins_b += 1
        if verbose:
            print(f"  game {g + 1}/{n_games}: A={wins_a} B={wins_b} D={draws}")
    return wins_a, wins_b, draws


def arena_parallel(path_a, path_b, config, n_games=20, num_workers=None):
    """Play n_games in parallel across subprocesses.

    Each subprocess loads its own network copies from disk so MPS tensors
    are never pickled.  Returns (wins_a, wins_b, draws).
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), n_games)

    config_dict = config.to_dict()
    work = []
    for g in range(n_games):
        swap = (g % 2 == 1)
        work.append((config_dict, path_a, path_b, swap))

    wins_a = wins_b = draws = 0
    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(_match_worker, w) for w in work]
        for f in as_completed(futures):
            result = f.result()
            if result is None:
                draws += 1
            elif result == 0:
                wins_a += 1
            else:
                wins_b += 1

    return wins_a, wins_b, draws
