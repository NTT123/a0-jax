"""Useful functions."""

import importlib
from functools import partial
from typing import Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import pax

from games.env import Enviroment as E

T = TypeVar("T")


@pax.pure
def batched_policy(agent, states):
    """Apply a policy to a batch of states.

    Also return the updated agent.
    """
    return agent, agent(states, batched=True)


def replicate(value: T, repeat: int) -> T:
    """Replicate along the first axis."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * repeat), value)


@pax.pure
def reset_env(env: E) -> E:
    """Return a reset enviroment."""
    env.reset()
    return env


@jax.jit
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


def select_tree(pred: jnp.ndarray, a, b):
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_util.tree_map(partial(jax.lax.select, pred), a, b)
