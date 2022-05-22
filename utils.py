"""Useful functions."""

import importlib
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


def import_class(path: str) -> E:
    """Import a class from a python file.

    For example:
    >> Game = import_class("connect_two_game.Connect2Game")

    Game is the Connect2Game class from `connection_two_game.py`.
    """
    names = path.split(".")
    mod_path, class_name = names[:-1], names[-1]
    mod = importlib.import_module(".".join(mod_path))
    return getattr(mod, class_name)
