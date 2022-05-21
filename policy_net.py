"""
A simple policy-value network
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pax


class PolicyValueNet(pax.Module):
    """
    Predict action probability and the value.

    Pi(s) returns the logit of actions to take at state s.
    V(s) returns the value of state s.
    """

    def __init__(self):
        super().__init__()
        self.action_head = pax.Sequential(
            pax.Linear(4, 128),
            jax.nn.relu,
            pax.Linear(128, 4),
        )
        self.value_head = pax.Sequential(
            pax.Linear(4, 128),
            jax.nn.relu,
            pax.Linear(128, 1),
        )

    def __call__(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Arguments:
            x: [batch_size, 4] the board state.

        Returns:
            (action_logits, value)
        """
        x = x.astype(jnp.float32)
        x = (x - 0.5) * 2.0  # normalize to [-1, 1]
        x = x[None, :]
        action_logits = self.action_head(x)[0]
        value = self.value_head(x)[0, 0]
        value = jax.nn.tanh(value)
        return action_logits, value
