"""
Monte Carlo tree search.
"""

import chex
import jax
import jax.numpy as jnp
import mctx

from utils import env_step


def recurrent_fn(params, rng_key: chex.Array, action: chex.Array, embedding):
    """One simulation step"""
    del rng_key
    agent = params
    env = embedding
    env, reward = jax.vmap(env_step)(env, action)
    state = env.canonical_observation()
    prior_logits, value = jax.vmap(lambda a, s: a(s), in_axes=(None, 0))(agent, state)
    invalid_actions = env.invalid_actions()
    assert invalid_actions.shape == prior_logits.shape
    prior_logits = jnp.where(invalid_actions, float("-inf"), prior_logits)
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
