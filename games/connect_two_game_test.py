from typing import List, NamedTuple

import chex
import jax.numpy as jnp
import pytest

from games.connect_two_game import Connect2Game, Connect2WinChecker


class CheckerTestData(NamedTuple):
    board: List[int]
    expected: int


checker_testdata = [
    CheckerTestData([1, 1, -1, 0], 1),
    CheckerTestData([0, 1, 1, -1], 1),
    CheckerTestData([0, -1, 1, 1], 1),
    CheckerTestData([1, -1, -1, 0], -1),
    CheckerTestData([0, 0, -1, -1], -1),
    CheckerTestData([1, -1, 0, 0], 0),
    CheckerTestData([0, 0, 0, 0], 0),
    CheckerTestData([1, 1, -1, -1], 0),
]


@pytest.mark.parametrize(["board", "expected"], checker_testdata)
def test_win_checker(board: List[int], expected: int) -> None:
    checker = Connect2WinChecker()
    assert checker(jnp.array(board)).item() == expected


def test_connect2_game_basics():
    game = Connect2Game()
    assert game.num_actions() == 4  # 4 positions
    game, _ = game.step(jnp.array(0))
    chex.assert_equal(game.board.tolist(), [1, 0, 0, 0])
    game, _ = game.step(jnp.array(1))
    chex.assert_equal(game.board.tolist(), [1, -1, 0, 0])
    game, _ = game.step(jnp.array(2))
    chex.assert_equal(game.board.tolist(), [1, -1, 1, 0])
    game, _ = game.step(jnp.array(3))
    chex.assert_equal(game.board.tolist(), [1, -1, 1, -1])
    assert game.terminated.item() is True


def test_connect2_game_reward_1():
    game = Connect2Game()
    game, r1 = game.step(jnp.array(1))
    assert r1.item() == 0
    game, r2 = game.step(jnp.array(0))
    assert r2.item() == 0
    game, r3 = game.step(jnp.array(2))
    assert r3.item() == 1
    assert game.terminated.item() is True


def test_connect2_game_reward_2():
    game = Connect2Game()
    game, _ = game.step(jnp.array(0))
    game, _ = game.step(jnp.array(1))
    game, _ = game.step(jnp.array(3))
    game, r4 = game.step(jnp.array(2))
    assert r4.item() == 1
    assert game.terminated.item() is True


def test_connect2_game_reward_3():
    game = Connect2Game()
    game, _ = game.step(jnp.array(0))
    game, r = game.step(jnp.array(0))
    assert r.item() == -1
    assert game.terminated.item() is True
