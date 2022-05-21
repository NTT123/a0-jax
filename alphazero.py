"""
AlphaZero functions
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import mctx
import pax

from env import Enviroment as E


@pax.pure
def reset_env(env: E) -> E:
    """Return a reset enviroment."""
    env.reset()
    return env


def step(env: E, action: chex.Array) -> Tuple[E, chex.Array]:
    """Execute one step in the enviroment."""
    env, reward = env.step(action)
    return env, reward


def replicate(value: chex.ArrayTree, repeat: int) -> chex.ArrayTree:
    """Replicate along the first axis."""
    return jax.tree_map(lambda x: jnp.stack([x] * repeat), value)


def recurrent_fn(params, rng_key: chex.Array, action: chex.Array, embedding):
    """One simulation step"""
    del rng_key
    agent = params
    env = embedding
    env, reward = jax.vmap(step)(env, action)
    state = env.canonical_observation()
    prior_logits, value = jax.vmap(lambda a, s: a(s), in_axes=(None, 0))(agent, state)
    assert env.board.shape == prior_logits.shape
    prior_logits = jnp.where(env.board == 0, prior_logits, float("-inf"))
    discount = -1.0 * jnp.ones_like(reward)
    assert discount.shape == env.terminated.shape
    # thank to the trick from
    # https://github.com/kenjyoung/mctx_learning_demo/blob/c0f396461b665e4c96227b98585a822c2e0d2e49/basic_tree_search.py#L95
    discount = jnp.where(env.terminated, 0.0, discount)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
        terminated=env.terminated,
    )
    return recurrent_fn_output, env
