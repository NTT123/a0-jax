"""Connect-Four game mechanics"""

from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pax

from games.env import Enviroment
from utils import select_tree


class Connect4WinChecker(pax.Module):
    """Check who won the game

    We use a conv2d for scanning the whole board.
    (We can do better by locating the recent move.)

    Filters to scan for winning patterns:

    1 0 0 0    1 1 1 1    1 0 0 0
    1 0 0 0    0 0 0 0    0 1 0 0
    1 0 0 0    0 0 0 0    0 0 1 0
    1 0 0 0    0 0 0 0    0 0 0 1

    0 0 0 1    0 0 0 0    0 0 0 1
    0 0 0 1    0 0 0 0    0 0 1 0
    0 0 0 1    0 0 0 0    0 1 0 0
    0 0 0 1    1 1 1 1    1 0 0 0
    """

    def __init__(self):
        super().__init__()
        conv = pax.Conv2D(1, 6, 4, padding="VALID")
        weight = np.zeros((4, 4, 1, 6), dtype=np.float32)
        weight[0, :, :, 0] = 1
        weight[:, 0, :, 1] = 1
        weight[-1, :, :, 2] = 1
        weight[:, -1, :, 3] = 1
        for i in range(4):
            weight[i, i, :, 4] = 1
            weight[i, 3 - i, :, 5] = 1
        assert weight.shape == conv.weight.shape
        self.conv = conv.replace(weight=weight)

    def __call__(self, board):
        board = board[None, :, :, None].astype(jnp.float32)
        x = self.conv(board)
        m = jnp.max(jnp.abs(x))
        m1 = jnp.where(m == jnp.max(x), 1, -1)
        return jnp.where(m == 4, m1, 0)


class Connect4Game(Enviroment):
    """Connect-Four game environment"""

    board: chex.Array
    who_play: chex.Array
    terminated: chex.Array
    winner: chex.Array
    num_cols: int
    num_rows: int

    def __init__(self, num_cols: int = 7, num_rows: int = 6):
        super().__init__()
        self.winner_checker = Connect4WinChecker()
        self.board = jnp.zeros((num_rows, num_cols), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.col_counts = jnp.zeros((num_cols,), dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.reset()

    def num_actions(self):
        return self.num_cols

    def invalid_actions(self) -> chex.Array:
        return self.col_counts >= self.num_rows

    def reset(self):
        self.board = jnp.zeros((self.num_rows, self.num_cols), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.col_counts = jnp.zeros((self.num_cols,), dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["Connect4Game", chex.Array]:
        """One step of the game.

        An invalid move will terminate the game with reward -1.
        """
        row_idx = self.col_counts[action]
        invalid_move = row_idx >= self.num_rows
        board_ = self.board.at[row_idx, action].set(self.who_play)
        self.board = select_tree(self.terminated, self.board, board_)
        self.winner = self.winner_checker(self.board)
        reward = self.winner * self.who_play
        # increase column counter
        self.col_counts = self.col_counts.at[action].set(self.col_counts[action] + 1)
        self.who_play = -self.who_play
        count = jnp.sum(self.col_counts)
        self.terminated = jnp.logical_or(self.terminated, reward != 0)
        self.terminated = jnp.logical_or(
            self.terminated, count >= self.num_cols * self.num_rows
        )
        self.terminated = jnp.logical_or(self.terminated, invalid_move)
        reward = jnp.where(invalid_move, -1.0, reward)
        return self, reward

    def render(self) -> None:
        """Render the game on screen."""
        for col in range(self.num_cols):
            print(col, end=" ")
        print()
        for row in reversed(range(self.num_rows)):
            for col in range(self.num_cols):
                if self.board[row, col].item() == 1:
                    print("X", end=" ")
                elif self.board[row, col].item() == -1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

    def observation(self) -> chex.Array:
        return self.board

    def canonical_observation(self) -> chex.Array:
        return self.board * self.who_play

    def is_terminated(self):
        return self.terminated

    def max_num_steps(self) -> int:
        return self.num_cols * self.num_rows

    def symmetries(self, state, action_weights):
        out = [(state, action_weights)]
        out.append((np.flip(state, axis=1), np.flip(action_weights)))
        return out


if __name__ == "__main__":
    game = Connect4Game()
    game.render()

    game, _ = game.step(6)
    game, _ = game.step(1)
    game, _ = game.step(1)
    game, _ = game.step(2)
    game, _ = game.step(6)
    game, _ = game.step(2)
    game, _ = game.step(2)
    game, _ = game.step(4)
    game, _ = game.step(6)
    game, _ = game.step(5)
    game, reward_ = game.step(6)
    game.render()
    print("Reward", reward_)
