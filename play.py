"""
Human vs AI play
"""

import pickle
import warnings

import jax.numpy as jnp
from fire import Fire

from env import Enviroment
from policy_net import PolicyValueNet
from utils import env_step, import_game, reset_env


def play_against_agent(
    agent: PolicyValueNet, env: Enviroment, human_first: bool = True
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
            logits, value = agent(s)
            logits = jnp.where(env.invalid_actions(), float("-inf"), logits)
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
    ckpt_filename: str = "./agent.ckpt",
    human_first: bool = True,
):
    """Load agent's weight from disk and start the game."""
    warnings.filterwarnings("ignore")
    env = import_game(game_class)()
    agent = PolicyValueNet()
    with open(ckpt_filename, "rb") as f:
        agent = agent.load_state_dict(pickle.load(f))
    play_against_agent(agent, env, human_first=human_first)


if __name__ == "__main__":
    Fire(main)
