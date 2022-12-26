"""Connect-Two game mechanics"""

from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pax

from games.env import Enviroment
from utils import select_tree


class Connect2WinChecker(pax.Module):
    """Check who won the game

    We use a conv1d for scanning the whole board.
    (We can do better by locating the recent move.)

    Result:
        +1: player p1 won
        -1: player p2 won
         0: undecided
    """

    def __init__(self):
        super().__init__()
        conv = pax.Conv1D(1, 1, 2, padding="VALID")
        weight = jnp.array([1.0, 1.0], dtype=conv.weight.dtype)
        weight = weight.reshape(conv.weight.shape)
        self.conv = conv.replace(weight=weight)

    def __call__(self, board: chex.Array) -> chex.Array:
        board = board[None, :, None].astype(jnp.float32)
        x = self.conv(board)
        is_p1_won: chex.Array = jnp.max(x) == 2  # 1 + 1
        is_p2_won: chex.Array = jnp.min(x) == -2  # -1 + -1
        return is_p1_won * 1 + is_p2_won * (-1)


class Connect2Game(Enviroment):
    """Connect-Two game environment"""

    board: chex.Array
    who_play: chex.Array
    count: chex.Array
    terminated: chex.Array
    winner: chex.Array

    def __init__(self):
        super().__init__()
        self.reset()
        self.winner_checker = Connect2WinChecker()
        self.board = jnp.zeros((4,), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)

    def num_actions(self) -> int:
        return 4

    def invalid_actions(self) -> chex.Array:
        return self.board != 0

    def reset(self):
        self.board = jnp.zeros((4,), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["Connect2Game", chex.Array]:
        """One step of the game.

        An invalid move will terminate the game with reward -1.
        """
        invalid_move = self.board[action] != 0
        board_ = self.board.at[action].set(self.who_play)
        self.board = select_tree(self.terminated, self.board, board_)
        self.winner = self.winner_checker(self.board)
        reward = self.winner * self.who_play
        self.who_play = -self.who_play
        self.count = self.count + 1
        self.terminated = jnp.logical_or(self.terminated, reward != 0)
        self.terminated = jnp.logical_or(self.terminated, self.count >= 4)
        self.terminated = jnp.logical_or(self.terminated, invalid_move)
        reward = jnp.where(invalid_move, -1.0, reward)
        return self, reward

    def render(self) -> None:
        """Render the game on screen."""
        print("0 1 2 3")
        N = len(self.board)
        for i in range(N):
            if self.board[i].item() == 1:
                print("X", end=" ")
            elif self.board[i].item() == -1:
                print("O", end=" ")
            else:
                print(".", end=" ")
        print()

    def observation(self) -> chex.Array:
        return self.board

    def canonical_observation(self) -> chex.Array:
        return self.board * self.who_play

    def is_terminated(self):
        return self.terminated

    def max_num_steps(self) -> int:
        return 4

    def symmetries(self, state, action_weights):
        return [(state, action_weights), (np.flip(state), np.flip(action_weights))]
