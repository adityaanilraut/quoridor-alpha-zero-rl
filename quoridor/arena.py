"""Pit two agents against each other to measure relative strength."""

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
