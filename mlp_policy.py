"""
A simple policy-value network
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pax


class MlpPolicyValueNet(pax.Module):
    """
    Predict action probability and the value.

    A(s) returns the logits of actions to take at state s.
    V(s) returns the value of state s.
    """

    def __init__(self, input_dims=(4,), num_actions=4):
        super().__init__()
        self.backbone = pax.Sequential(
            pax.Linear(np.prod(input_dims), 128), jax.nn.relu
        )
        self.action_head = pax.Sequential(
            pax.Linear(128, 128), jax.nn.relu, pax.Linear(128, num_actions)
        )
        self.value_head = pax.Sequential(
            pax.Linear(128, 128), jax.nn.relu, pax.Linear(128, 1)
        )
        self.input_dims = input_dims

    def __call__(
        self, x: chex.Array, batched: bool = False
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Arguments:
            x: the board state. [batch_size, D]  or [D]

        Returns:
            (action_logits, value)
        """
        x = x.astype(jnp.float32)
        if not batched:
            x = x[None]  # add batch dimension
        x = jnp.reshape(x, (x.shape[0], -1))  # flatten before input mlp
        x = self.backbone(x)
        action_logits = self.action_head(x)
        value = jax.nn.tanh(self.value_head(x))
        if batched:
            return action_logits[:, :], value[:, 0]
        else:
            return action_logits[0, :], value[0, 0]
