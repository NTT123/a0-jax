"""
Human vs AI play
"""

import pickle
import random
import warnings
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from fire import Fire

from games.env import Enviroment
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env


class PlayResults(NamedTuple):
    win_count: chex.Array
    draw_count: chex.Array
    loss_count: chex.Array


@partial(
    jax.jit,
    static_argnames=("num_simulations", "disable_mcts", "random_action"),
)
def play_one_move(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    disable_mcts: bool = False,
    num_simulations: int = 1024,
    random_action: bool = True,
):
    """Play a move using agent's policy"""
    if disable_mcts:
        action_logits, value = agent(env.canonical_observation())
        action_weights = jax.nn.softmax(action_logits, axis=-1)
    else:
        batched_env: Enviroment = replicate(env, 1)  # type: ignore
        rng_key, rng_key_1 = jax.random.split(rng_key)  # type: ignore
        policy_output = improve_policy_with_mcts(
            agent,
            batched_env,
            rng_key_1,  # type: ignore
            rec_fn=recurrent_fn,
            num_simulations=num_simulations,
        )
        action_weights = policy_output.action_weights[0]
        root_idx = policy_output.search_tree.ROOT_INDEX
        value = policy_output.search_tree.node_values[0, root_idx]

    if random_action:
        action = jax.random.categorical(rng_key, jnp.log(action_weights), axis=-1)
    else:
        action = jnp.argmax(action_weights)
    return action, action_weights, value


def agent_vs_agent(
    agent1,
    agent2,
    env: Enviroment,
    rng_key: chex.Array,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
):
    """A game of agent1 vs agent2."""

    def cond_fn(state):
        env, step = state[0], state[-1]
        # pylint: disable=singleton-comparison
        not_ended = env.is_terminated() == False
        not_too_long = step <= env.max_num_steps()
        return jnp.logical_and(not_ended, not_too_long)

    def loop_fn(state):
        env, a1, a2, _, rng_key, turn, step = state
        rng_key_1, rng_key = jax.random.split(rng_key)
        action, _, _ = play_one_move(
            a1,
            env,
            rng_key_1,
            disable_mcts=disable_mcts,
            num_simulations=num_simulations_per_move,
        )
        env, reward = env_step(env, action)
        state = (env, a2, a1, turn * reward, rng_key, -turn, step + 1)
        return state

    state = (
        reset_env(env),
        agent1,
        agent2,
        jnp.array(0),
        rng_key,
        jnp.array(1),
        jnp.array(1),
    )
    state = jax.lax.while_loop(cond_fn, loop_fn, state)
    return state[3]


@partial(jax.jit, static_argnums=(4, 5, 6))
def agent_vs_agent_multiple_games(
    agent1,
    agent2,
    env,
    rng_key,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
    num_games: int = 128,
) -> PlayResults:
    """Fast agent vs agent evaluation."""
    rng_key_list = jax.random.split(rng_key, num_games)
    rng_keys = jnp.stack(rng_key_list, axis=0)  # type: ignore
    avsa = partial(
        agent_vs_agent,
        disable_mcts=disable_mcts,
        num_simulations_per_move=num_simulations_per_move,
    )
    batched_avsa = jax.vmap(avsa, in_axes=(None, None, 0, 0))
    envs = replicate(env, num_games)
    results = batched_avsa(agent1, agent2, envs, rng_keys)
    win_count = jnp.sum(results == 1)
    draw_count = jnp.sum(results == 0)
    loss_count = jnp.sum(results == -1)
    return PlayResults(
        win_count=win_count, draw_count=draw_count, loss_count=loss_count
    )


def human_vs_agent(
    agent,
    env: Enviroment,
    human_first: bool = True,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
):
    """A game of human vs agent."""
    env = reset_env(env)
    agent_turn = 1 if human_first else 0
    rng_key = jax.random.PRNGKey(random.randint(0, 999999))
    for i in range(1000):
        print()
        print(f"Move {i}")
        print("======")
        print()
        env.render()
        if i % 2 == agent_turn:
            print()
            s = env.canonical_observation()
            print("#  s =", s)
            rng_key_1, rng_key = jax.random.split(rng_key)
            action, action_weights, value = play_one_move(
                agent,
                env,
                rng_key_1,
                disable_mcts=disable_mcts,
                num_simulations=num_simulations_per_move,
                random_action=False,
            )
            print("#  A(s) =", action_weights)
            print("#  V(s) =", value)
            env, reward = env_step(env, action)
            print(f"#  Agent selected action {action}, got reward {reward}")
        else:
            action = input("> ")
            action = env.parse_action(action)
            env, reward = env_step(env, jnp.array(action, dtype=jnp.int32))
            print(f"#  Human selected action {action}, got reward {reward}")
        if env.is_terminated().item():
            break
    else:
        print("Timeout!")
    print()
    print("Final board")
    print("===========")
    print()
    env.render()
    print()


def main(
    game_class: str = "games.connect_two_game.Connect2Game",
    agent_class="policies.mlp_policy.MlpPolicyValueNet",
    ckpt_filename: str = "./agent.ckpt",
    human_first: bool = False,
    disable_mcts: bool = False,
    num_simulations_per_move: int = 128,
):
    """Load agent's weight from disk and start the game."""
    warnings.filterwarnings("ignore")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    human_vs_agent(
        agent,
        env,
        human_first=human_first,
        disable_mcts=disable_mcts,
        num_simulations_per_move=num_simulations_per_move,
    )


if __name__ == "__main__":
    Fire(main)
