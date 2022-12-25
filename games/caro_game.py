"""Caro (Gomoku) game mechanics


Implement Pro rule. Reference: http://gomokuworld.com/gomoku/2
"""

from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pax

from games.env import Enviroment
from utils import select_tree


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
    num_cols: int
    num_rows: int
    count: chex.Array

    def __init__(self, num_cols: int = 9, num_rows: int = 9, pro_rule_dist: int = 3):
        super().__init__()
        assert num_cols % 2 == 1 and num_rows % 2 == 1
        assert pro_rule_dist in [3, 4]
        self.pro_rule_dist = pro_rule_dist
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.winner_checker = CaroWinnerChecker()
        self.reset()

    def num_actions(self):
        return self.num_cols * self.num_rows

    def invalid_actions(self) -> chex.Array:
        return self.board != 0

    def reset(self):
        self.board = jnp.zeros((self.num_rows * self.num_cols), dtype=jnp.int32)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.count = jnp.array(0, dtype=jnp.int32)

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["CaroGame", chex.Array]:
        """One step of the game.

        An invalid move will terminate the game with reward -1.
        """
        i, j = jnp.divmod(action, self.num_cols)
        mid_i = self.num_rows // 2
        mid_j = self.num_cols // 2
        d_i = jnp.abs(mid_i - i)
        d_j = jnp.abs(mid_j - j)
        not_at_center = jnp.logical_or(d_i != 0, d_j != 0)
        near_center = jnp.logical_and(
            d_i < self.pro_rule_dist, d_j < self.pro_rule_dist
        )
        is_first_move = self.count == 0
        is_third_move = self.count == 2
        invalid_first_move = jnp.logical_and(is_first_move, not_at_center)
        invalid_third_move = jnp.logical_and(is_third_move, near_center)
        invalid_move = jnp.logical_or(invalid_first_move, invalid_third_move)
        invalid_move = jnp.logical_or(invalid_move, self.board[action] != 0)
        board_ = self.board.at[action].set(self.who_play)
        self.board = select_tree(self.terminated, self.board, board_)
        winner = self.winner_checker(self.observation())
        reward_ = winner * self.who_play
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
        print(end="  ")
        for col in range(self.num_cols):
            print(chr(ord("a") + col), end=" ")
        print()
        for row in range(self.num_rows):
            print(chr(ord("a") + row), end=" ")
            for col in range(self.num_cols):
                if board[row, col].item() == 1:
                    print("X", end=" ")
                elif board[row, col].item() == -1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print()
        print(end="  ")
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

    def parse_action(self, action_str: str) -> int:
        sa, sb = list(action_str.strip().replace(" ", ""))
        a = ord(sa) - ord("a")
        b = ord(sb) - ord("a")
        return a * self.num_cols + b


class CaroGame11x11(CaroGame):
    """Caro game with board size 11x11"""

    def __init__(self):
        super().__init__(num_cols=11, num_rows=11, pro_rule_dist=3)


class CaroGame13x13(CaroGame):
    """Caro game with board size 13x13"""

    def __init__(self):
        super().__init__(num_cols=13, num_rows=13, pro_rule_dist=3)


class CaroGame15x15(CaroGame):
    """Caro game with board size 15x15"""

    def __init__(self):
        super().__init__(num_cols=15, num_rows=15, pro_rule_dist=4)


if __name__ == "__main__":
    game = CaroGame()
    while not game.is_terminated().item():
        game.render()
        user_input: str = input("> ")
        user_action = game.parse_action(user_input)
        game, reward = game.step(user_action)

    print("Final board")
    game.render()
