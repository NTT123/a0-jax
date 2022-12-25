"""Tic tac toe game mechanics"""

from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pax

from games.env import Enviroment
from utils import select_tree


class TicTacToeWinnerChecker(pax.Module):
    """Check who won the game

    We use a conv2d for scanning the whole board.
    (We can do better by locating the recent move.)

    Filters to scan for winning patterns:

    1 0 0    0 1 0    0 0 1    1 0 0
    1 0 0    0 1 0    0 0 1    0 1 0
    1 0 0    0 1 0    0 0 1    0 0 1

    1 1 1    0 0 0    0 0 0    0 0 1
    0 0 0    1 1 1    0 0 0    0 1 0
    0 0 0    0 0 0    1 1 1    1 0 0

    """

    def __init__(self):
        super().__init__()
        conv = pax.Conv2D(1, 8, 3, padding="VALID")
        weight = np.zeros((3, 3, 1, 8), dtype=np.float32)
        weight[0, :, :, 0] = 1
        weight[1, :, :, 1] = 1
        weight[2, :, :, 2] = 1
        weight[:, 0, :, 3] = 1
        weight[:, 1, :, 4] = 1
        weight[:, 2, :, 5] = 1
        for i in range(3):
            weight[i, i, :, 6] = 1
            weight[i, 2 - i, :, 7] = 1
        assert weight.shape == conv.weight.shape
        self.conv = conv.replace(weight=weight)

    def __call__(self, board):
        board = board[None, :, :, None].astype(jnp.float32)
        x = self.conv(board)
        m = jnp.max(jnp.abs(x))
        m1 = jnp.where(m == jnp.max(x), 1, -1)
        return jnp.where(m == 3, m1, 0)


class TicTacToeGame(Enviroment):
    """Tic-tac-toe game environment"""

    board: chex.Array
    who_play: chex.Array
    terminated: chex.Array
    winner: chex.Array
    num_cols: int = 3
    num_rows: int = 3

    def __init__(self, num_cols: int = 3, num_rows: int = 3):
        super().__init__()
        self.winner_checker = TicTacToeWinnerChecker()
        self.board = jnp.zeros((num_rows * num_cols,), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.reset()

    def num_actions(self):
        return self.num_cols * self.num_rows

    def invalid_actions(self) -> chex.Array:
        return self.board != 0

    def reset(self):
        self.board = jnp.zeros((self.num_rows * self.num_cols,), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["TicTacToeGame", chex.Array]:
        """One step of the game.

        An invalid move will terminate the game with reward -1.
        """
        invalid_move = self.board[action] != 0
        board_ = self.board.at[action].set(self.who_play)
        self.board = select_tree(self.terminated, self.board, board_)
        self.winner = self.winner_checker(self.observation())
        reward_ = self.winner * self.who_play
        # increase column counter
        self.who_play = -self.who_play
        self.count = self.count + 1
        self.terminated = jnp.logical_or(self.terminated, reward_ != 0)
        self.terminated = jnp.logical_or(
            self.terminated, self.count >= self.num_cols * self.num_rows
        )
        self.terminated = jnp.logical_or(self.terminated, invalid_move)
        reward_ = jnp.where(invalid_move, -1.0, reward_)
        return self, reward_

    def render(self) -> None:
        """Render the game on screen."""
        board = self.observation()
        for row in reversed(range(self.num_rows)):
            for col in range(self.num_cols):
                if board[row, col].item() == 1:
                    print("X", end=" ")
                elif board[row, col].item() == -1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
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

    def symmetries(self, state, action_weights):
        action = action_weights.reshape((self.num_rows, self.num_cols))
        out = []
        for rotate in range(4):
            rotated_state = np.rot90(state, rotate, axes=(0, 1))
            rotated_action = np.rot90(action, rotate, axes=(0, 1))
            out.append((rotated_state, rotated_action.reshape((-1,))))

            flipped_state = np.fliplr(rotated_state)
            flipped_action = np.fliplr(rotated_action)
            out.append((flipped_state, flipped_action.reshape((-1,))))
        return out


if __name__ == "__main__":
    game = TicTacToeGame()
    game.render()
    game, reward = game.step(8)
    game, reward = game.step(6)
    game, reward = game.step(5)
    game, reward = game.step(4)
    game, reward = game.step(2)
    game.render()
    print("Reward", reward)
