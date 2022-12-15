"""
Enviroment base class.
"""

from typing import Any, List, Tuple, TypeVar

import chex
import pax

E = TypeVar("E")


class Enviroment(pax.Module):
    """A template for environments."""

    def __init__(self):
        super().__init__()

    def step(self: E, action: chex.Array) -> Tuple[E, chex.Array]:
        """A single env step."""
        raise NotImplementedError()

    def reset(self):
        """Reset the enviroment."""

    def is_terminated(self) -> chex.Array:
        """The env is terminated."""
        raise NotImplementedError()

    def observation(self) -> Any:
        """The observation from env."""

    def canonical_observation(self) -> Any:
        """Return the canonical observation."""

    def num_actions(self) -> int:
        """Return the size of the action space."""
        raise NotImplementedError()

    def invalid_actions(self) -> chex.Array:
        """An boolean array indicating invalid actions.

        Returns:
            invalid_action: the i-th element is true if action `i` is invalid. [num_actions].
        """
        raise NotImplementedError()

    def max_num_steps(self) -> int:
        """Return the maximum number of steps until the game is terminated."""
        raise NotImplementedError()

    def parse_action(self, action_str: str) -> int:
        """Parse a string action and return a number."""
        return int(action_str)

    def symmetries(self, state, action_weights) -> List[Tuple[chex.Array, ...]]:
        """The default symmetry group is the identity.

        We use this method for data augmentation in training.
        """
        return [(state, action_weights)]
