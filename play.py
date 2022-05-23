"""
Human vs AI play
"""

import pickle
import warnings

import jax
import jax.numpy as jnp
from fire import Fire

from env import Enviroment
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env


def play_against_agent(
    agent,
    env: Enviroment,
    human_first: bool = True,
    enable_mcts: bool = False,
    num_simulations_per_move: int = 128,
):
    """A game of human vs agent."""
    env = reset_env(env)
    agent_turn = 1 if human_first else 0
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
            if enable_mcts:
                batched_env = replicate(env, 1)
                policy_output = improve_policy_with_mcts(
                    agent,
                    batched_env,
                    recurrent_fn,
                    jax.random.PRNGKey(i),
                    num_simulations_per_move,
                )
                logits = jnp.log(policy_output.action_weights)
                root_idx = policy_output.search_tree.ROOT_INDEX
                value = policy_output.search_tree.node_values[0, root_idx]
            else:
                logits, value = agent(s)
            # the network should be able to learn to avoid invalid actions
            # logits = jnp.where(env.invalid_actions(), float("-inf"), logits)
            print("#  A(s) =", logits)
            print("#  V(s) =", value)
            action = jnp.argmax(logits, axis=-1).item()
            env, reward = env_step(env, action)
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
    agent_class="mlp_policy_net.MlpPolicyValueNet",
    ckpt_filename: str = "./agent.ckpt",
    human_first: bool = True,
    enable_mcts: bool = False,
    num_simulations_per_move: bool = 128,
):
    """Load agent's weight from disk and start the game."""
    warnings.filterwarnings("ignore")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f))
    agent = agent.eval()
    play_against_agent(
        agent,
        env,
        human_first=human_first,
        enable_mcts=enable_mcts,
        num_simulations_per_move=num_simulations_per_move,
    )


if __name__ == "__main__":
    Fire(main)
