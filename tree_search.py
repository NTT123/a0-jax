"""
Monte Carlo tree search.
"""

import chex
import jax
import jax.numpy as jnp
import mctx

from env import Enviroment as E
from utils import batched_policy, env_step


def recurrent_fn(params, rng_key: chex.Array, action: chex.Array, embedding):
    """One simulation step in MCTS."""
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
    terminated = env.is_terminated()
    assert value.shape == terminated.shape
    value = jnp.where(terminated, 0.0, value)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
        terminated=terminated,
    )
    return recurrent_fn_output, env


def improve_policy_with_mcts(
    agent, env: E, rec_fn, rng_key: chex.Array, num_simulations: int
):
    """Improve agent policy using MCTS.

    Returns:
        An improved policy.
    """
    state = env.canonical_observation()
    _, (prior_logits, value) = batched_policy(agent, state)
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=env)
    policy_output = mctx.gumbel_muzero_policy(
        params=agent,
        rng_key=rng_key,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=env.num_actions(),
        invalid_actions=env.invalid_actions(),
        qtransform=mctx.qtransform_by_parent_and_siblings,
        gumbel_scale=1.0,
    )
    return policy_output
