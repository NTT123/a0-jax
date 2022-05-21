"""Connect-Two game mechanics"""

from typing import Tuple

import chex
import jax.numpy as jnp
import jmp
import pax

from env import Enviroment


class Connect2WinChecker(pax.Module):
    """Check who won the game

    We use a conv1d for scanning the whole board.
    (We can do better by locating the recent move.)
    """

    def __init__(self):
        super().__init__()
        conv = pax.Conv1D(1, 1, 2, padding="VALID")
        weight = jnp.array([1.0, 1.0], dtype=conv.weight.dtype)
        weight = weight.reshape(conv.weight.shape)
        self.conv = conv.replace(weight=weight)

    def __call__(self, board):
        board = board[None, :, None].astype(jnp.float32)
        x = self.conv(board)
        m = jnp.max(jnp.abs(x))
        m1 = jnp.where(m == jnp.max(x), 1, -1)
        return jnp.where(m == 2, m1, 0)


class Connect2Game(Enviroment):
    """Connect2 game enviroment"""

    board: chex.Array
    who_play: chex.Array
    count: chex.Array
    terminated: chex.Array
    win: chex.Array

    def __init__(self):
        super().__init__()
        self.reset()
        self.win_checker = Connect2WinChecker()
        self.board = jnp.zeros((4,), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.win = jnp.array(0, dtype=jnp.int32)

    def reset(self):
        self.board = jnp.zeros((4,), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.win = jnp.array(0, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["Connect2Game", chex.Array]:
        board_ = self.board.at[action].set(self.who_play)
        self.board = jmp.select_tree(self.terminated, self.board, board_)
        self.win = self.win_checker(self.board)
        reward = self.win * self.who_play
        self.who_play = -self.who_play
        self.count = self.count + 1
        self.terminated = jnp.logical_or(self.terminated, reward != 0)
        self.terminated = jnp.logical_or(self.terminated, self.count >= 4)
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
        return self.board * self.who_play[..., None]

    def is_terminated(self):
        return self.terminated
