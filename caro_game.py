"""Caro (Gomoku) game mechanics"""

from typing import Tuple

import chex
import jax.numpy as jnp
import jmp
import numpy as np
import pax

from env import Enviroment


class CaroWinnerChecker(pax.Module):
    """Check who won the game

    We use a conv2d for scanning the whole board.
    (We can do better by locating the recent move.)
    """

    def __init__(self):
        super().__init__()
        conv = pax.Conv2D(1, 6, 5, padding="VALID")
        weight = np.zeros((5, 5, 1, 6), dtype=np.float32)
        weight[0, :, :, 0] = 1
        weight[:, 0, :, 1] = 1
        weight[-1, :, :, 2] = 1
        weight[:, -1, :, 3] = 1
        for i in range(5):
            weight[i, i, :, 4] = 1
            weight[i, 4 - i, :, 5] = 1
        assert weight.shape == conv.weight.shape
        self.conv = conv.replace(weight=weight)

    def __call__(self, board):
        board = board[None, :, :, None].astype(jnp.float32)
        x = self.conv(board)
        m = jnp.max(jnp.abs(x))
        m1 = jnp.where(m == jnp.max(x), 1, -1)
        return jnp.where(m == 5, m1, 0)


class CaroGame(Enviroment):
    """Caro game environment"""

    board: chex.Array
    who_play: chex.Array
    terminated: chex.Array
    winner: chex.Array
    num_cols: int
    num_rows: int
    count: chex.Array

    def __init__(self, num_cols: int = 9, num_rows: int = 9):
        super().__init__()
        self.num_rows = num_rows
        self.winner_checker = CaroWinnerChecker()
        self.board = jnp.zeros((num_rows * num_cols,), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.num_cols = num_cols
        self.reset()

    def num_actions(self):
        return self.num_cols * self.num_rows

    def invalid_actions(self) -> chex.Array:
        return self.board != 0

    def reset(self):
        assert self.num_rows % 2 == 1 and self.num_cols % 2 == 1
        self.board = jnp.zeros((self.num_rows * self.num_cols,), dtype=jnp.int32)
        self.board = self.board.at[self.num_rows * self.num_cols // 2].set(1)
        self.who_play = jnp.array(-1, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["CaroGame", chex.Array]:
        """One step of the game.

        An invalid move will terminate the game with reward -1.
        """
        invalid_move = self.board[action] != 0
        board_ = self.board.at[action].set(self.who_play)
        self.board = jmp.select_tree(self.terminated, self.board, board_)
        self.winner = self.winner_checker(self.observation())
        reward_ = self.winner * self.who_play
        self.who_play = -self.who_play
        self.count = self.count + 1
        self.terminated = jnp.logical_or(self.terminated, reward_ != 0)
        self.terminated = jnp.logical_or(
            self.terminated, self.count >= self.num_cols * self.num_rows
        )
        self.terminated = jnp.logical_or(self.terminated, invalid_move)
        reward_ = jnp.where(invalid_move, -1.0, reward_)
        return self, reward_

    def step_xy(self, x: int, y: int):
        """step function with 2d actions."""
        return self.step(y * self.num_cols + x)

    def render(self) -> None:
        """Render the game on screen."""
        board = self.observation()
        for row in reversed(range(self.num_rows)):
            print(row, end=" ")
            for col in range(self.num_cols):
                if board[row, col].item() == 1:
                    print("X", end=" ")
                elif board[row, col].item() == -1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
        print(end="  ")
        for col in range(self.num_cols):
            print(col, end=" ")
        print()

    def observation(self) -> chex.Array:
        board = self.board
        return jnp.reshape(board, board.shape[:-1] + (self.num_rows, self.num_cols))

    def canonical_observation(self) -> chex.Array:
        return self.observation() * self.who_play

    def is_terminated(self):
        return self.terminated

    def max_num_steps(self) -> int:
        return self.num_cols * self.num_rows


if __name__ == "__main__":
    game = CaroGame()
    game.render()
    game, reward = game.step_xy(3, 1)
    game, reward = game.step_xy(2, 4)
    game, reward = game.step_xy(3, 3)
    game, reward = game.step_xy(3, 4)
    game, reward = game.step_xy(3, 5)
    game, reward = game.step_xy(1, 4)
    game, reward = game.step_xy(3, 2)
    game, reward = game.step_xy(5, 4)
    game.render()
    print("Reward", reward)
