"""
Human vs AI play
"""

import pickle
import random
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
from fire import Fire

from env import Enviroment
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env


def _apply_temperature(logits, temperature):
    """Returns `logits / temperature`, supporting also temperature=0."""
    # The max subtraction prevents +inf after dividing by a small temperature.
    logits = logits - jnp.max(logits, keepdims=True, axis=-1)
    tiny = jnp.finfo(logits.dtype).tiny
    return logits / jnp.maximum(tiny, temperature)


@partial(jax.jit, static_argnames=("temperature", "num_simulations", "enable_mcts"))
def play_one_move(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    enable_mcts: bool = False,
    num_simulations: int = 1024,
    temperature=0.2,
):
    """Play a move using agent's policy"""
    if enable_mcts:
        batched_env = replicate(env, 1)
        policy_output = improve_policy_with_mcts(
            agent,
            batched_env,
            rng_key,
            rec_fn=recurrent_fn,
            num_simulations=num_simulations,
            temperature=temperature,
        )
        action = policy_output.action
        action_weights = jnp.log(policy_output.action_weights)
        root_idx = policy_output.search_tree.ROOT_INDEX
        value = policy_output.search_tree.node_values[0, root_idx]
    else:
        action_logits, value = agent(env.canonical_observation())
        action_logits_ = _apply_temperature(action_logits, temperature)
        action_weights = jax.nn.softmax(action_logits_, axis=-1)
        action = jax.random.categorical(rng_key, action_logits)

    return action, action_weights, value


def agent_vs_agent(
    agent1,
    agent2,
    env: Enviroment,
    rng_key: chex.Array,
    enable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
    temperature: float = 0.2,
):
    """A game of agent1 vs agent2."""
    env = reset_env(env)
    agents = [agent1, agent2]
    turn = 1
    for i in range(1000):
        agent = agents[i % 2]
        rng_key_1, rng_key = jax.random.split(rng_key)
        action, _, _ = play_one_move(
            agent,
            env,
            rng_key_1,
            enable_mcts=enable_mcts,
            num_simulations=num_simulations_per_move,
            temperature=temperature,
        )
        env, reward = env_step(env, action.item())
        if env.is_terminated().item():
            # return reward from agent1 point of view
            return turn * reward
        turn = -turn


def agent_vs_agent_multiple_games(
    agent1,
    agent2,
    env,
    enable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
    temperature: float = 0.2,
    num_games: int = 128,
):
    win_count, draw_count, loss_count = 0, 0, 0
    rng_keys = jax.random.split(
        jax.random.PRNGKey(random.randint(0, 9999999)), num_games
    )
    for i in range(num_games):
        result = agent_vs_agent(
            agent1,
            agent2,
            env,
            rng_keys[i],
            enable_mcts,
            num_simulations_per_move,
            temperature,
        )
        if result == 1:
            win_count += 1
        elif result == -1:
            loss_count += 1
        else:
            draw_count += 1
    return win_count, draw_count, loss_count


def human_vs_agent(
    agent,
    env: Enviroment,
    human_first: bool = True,
    enable_mcts: bool = False,
    num_simulations_per_move: int = 1024,
    temperature: float = 0.2,
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
                enable_mcts=enable_mcts,
                num_simulations=num_simulations_per_move,
                temperature=temperature,
            )
            print("#  A(s) =", action_weights)
            print("#  V(s) =", value)
            env, reward = env_step(env, action.item())
            print(f"#  Agent selected action {action}, got reward {reward}")
        else:
            action = int(input("> "))
            env, reward = env_step(env, action)
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
    game_class: str = "connect_two_game.Connect2Game",
    agent_class="mlp_policy.MlpPolicyValueNet",
    ckpt_filename: str = "./agent.ckpt",
    human_first: bool = False,
    enable_mcts: bool = False,
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
        enable_mcts=enable_mcts,
        num_simulations_per_move=num_simulations_per_move,
    )


if __name__ == "__main__":
    Fire(main)
