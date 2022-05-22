"""
A simple policy-value network
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pax


class MlpPolicyValueNet(pax.Module):
    """
    Predict action probability and the value.

    A(s) returns the logits of actions to take at state s.
    V(s) returns the value of state s.
    """

    def __init__(self, input_dims=(4,), num_actions=4):
        super().__init__()
        self.backbone = pax.Sequential(pax.Linear(input_dims[0], 128), jax.nn.relu)
        self.action_head = pax.Sequential(
            pax.Linear(128, 128), jax.nn.relu, pax.Linear(128, num_actions)
        )
        self.value_head = pax.Sequential(
            pax.Linear(128, 128), jax.nn.relu, pax.Linear(128, 1)
        )

    def __call__(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Arguments:
            x: [batch_size, D] the board state.

        Returns:
            (action_logits, value)
        """
        x = x.astype(jnp.float32)
        # flatten and create the batch dimension [1, D]
        x = jnp.reshape(x, (1, -1))
        x = self.backbone(x)
        action_logits = self.action_head(x)[0]
        value = self.value_head(x)[0, 0]
        value = jax.nn.tanh(value)
        return action_logits, value
