"""
Plot MCTS tree.


A modified version on https://github.com/deepmind/mctx/blob/main/examples/visualization_demo.py

Usage:
    # install dependencies (on Ubuntu)
    sudo apt-get install graphviz graphviz-dev -y
    pip install pygraphviz

    # run
    python plot_search_tree.py
    # ./search_tree.png
"""

import os
import pickle

import jax
import jax.numpy as jnp
import pygraphviz
from fire import Fire

from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import import_class, replicate


def main(
    game_class: str = "games.connect_two_game.Connect2Game",
    agent_class="policies.mlp_policy.MlpPolicyValueNet",
    ckpt_filepath: str = "./agent.ckpt",
    num_simulations: int = 32,
):
    """Run a `muzero_policy` at the start position and plot the search tree."""
    batch_size = 1
    game = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=game.observation().shape,
        num_actions=game.num_actions(),
    )
    if os.path.isfile(ckpt_filepath):
        print("Loading checkpoint at", ckpt_filepath)
        with open(ckpt_filepath, "rb") as f:
            agent = agent.load_state_dict(pickle.load(f)["agent"])
    agent = agent.eval()
    game = replicate(game, batch_size)
    rng_key: jnp.ndarray = jax.random.PRNGKey(42)  # type: ignore
    policy_output = improve_policy_with_mcts(
        agent, game, rng_key, recurrent_fn, num_simulations
    )
    tree = policy_output.search_tree

    def node_to_str(node_i, reward=0, discount=1):
        return (
            f"{node_i}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n"
        )

    def edge_to_str(node_i, a_i):
        node_index = jnp.full([batch_size], node_i)
        probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
        return (
            f"a{a_i}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"
            f"p: {probs[a_i]:.2f}\n"
        )

    graph = pygraphviz.AGraph(directed=True)
    batch_index = 0
    # Add root
    graph.add_node(0, label=node_to_str(node_i=0), color="green")
    # Add all other nodes and connect them up.
    for node_i in range(tree.num_simulations):
        for a_i in range(tree.num_actions):
            # Index of children, or -1 if not expanded
            children_i = tree.children_index[batch_index, node_i, a_i]
            if children_i >= 0:
                graph.add_node(
                    children_i,
                    label=node_to_str(
                        node_i=children_i,
                        reward=tree.children_rewards[batch_index, node_i, a_i],
                        discount=tree.children_discounts[batch_index, node_i, a_i],
                    ),
                    color="red",
                )
                graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

    graph.draw("search_tree.png", prog="dot")


if __name__ == "__main__":
    Fire(main)
