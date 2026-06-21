"""Regression tests for arena win-attribution. Run: python -m pytest tests/test_arena.py"""

from quoridor.arena import _result_code


def test_result_code_no_swap():
    # agent_a is player 0, agent_b is player 1
    assert _result_code(0, swap=False) == 0      # player 0 (a) won
    assert _result_code(1, swap=False) == 1      # player 1 (b) won
    assert _result_code(None, swap=False) is None


def test_result_code_swapped():
    # agent_a is player 1, agent_b is player 0 -> indices invert
    assert _result_code(1, swap=True) == 0       # player 1 (a) won -> a-win
    assert _result_code(0, swap=True) == 1       # player 0 (b) won -> b-win
    assert _result_code(None, swap=True) is None


def test_swap_does_not_flip_winner():
    """The same agent winning must yield the same result code regardless of
    which colour it was assigned (this is exactly what the gating bug broke)."""
    # agent_a wins: as player 0 when not swapped, as player 1 when swapped.
    assert _result_code(0, swap=False) == _result_code(1, swap=True) == 0
    # agent_b wins: as player 1 when not swapped, as player 0 when swapped.
    assert _result_code(1, swap=False) == _result_code(0, swap=True) == 1


if __name__ == "__main__":
    test_result_code_no_swap()
    test_result_code_swapped()
    test_swap_does_not_flip_winner()
    print("ok: arena attribution")
