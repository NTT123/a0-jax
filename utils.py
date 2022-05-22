"""Useful functions."""


from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pax

from env import Enviroment as E


def batched_policy(agent, states):
    """Apply a policy to a batch of states."""
    policy_fn = jax.vmap(lambda a, s: a(s), in_axes=(None, 0))
    return policy_fn(agent, states)


def replicate(value: chex.ArrayTree, repeat: int) -> chex.ArrayTree:
    """Replicate along the first axis."""
    return jax.tree_map(lambda x: jnp.stack([x] * repeat), value)


@pax.pure
def reset_env(env: E) -> E:
    """Return a reset enviroment."""
    env.reset()
    return env


def env_step(env: E, action: chex.Array) -> Tuple[E, chex.Array]:
    """Execute one step in the enviroment."""
    env, reward = env.step(action)
    return env, reward
