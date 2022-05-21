"""
Plot MCTS tree.


A modified version on https://github.com/deepmind/mctx/blob/main/examples/visualization_demo.py

Usage:
    # install dependencies (on Ubuntu)
    sudo apt-get install graphviz graphviz-dev -y
    pip install pygraphviz

    # run
    python plot_search_tree.py
    # output is at ./search_search.png
"""

import jax
import jax.numpy as jnp
import mctx
import pygraphviz

from connect_two_game import Connect2Game
from policy_net import PolicyValueNet
from tree_search import recurrent_fn
from utils import replicate

agent = PolicyValueNet()
batch_size = 1
game = Connect2Game()
prior_logits, value = agent(game.board)
root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=game)
root = replicate(root, batch_size)
rng_key = jax.random.PRNGKey(42)

policy_output = mctx.gumbel_muzero_policy(
    params=agent,
    rng_key=rng_key,
    root=root,
    recurrent_fn=recurrent_fn,
    num_simulations=1280,
    gumbel_scale=1.0,
    qtransform=mctx.qtransform_by_parent_and_siblings,
)

tree = policy_output.search_tree
action_labels = ["a0", "a1", "a2", "a3"]


def node_to_str(node_i, reward=0, discount=1):
    return (
        f"{node_i}\n"
        f"Reward: {reward:.2f}\n"
        f"Discount: {discount:.2f}\n"
        f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
        f"Visits: {tree.node_visits[batch_index, node_i]}\n"
        f"Terminated: {tree.children_terminated[batch_index, node_i]}\n"
    )


def edge_to_str(node_i, a_i):
    node_index = jnp.full([batch_size], node_i)
    probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
    return (
        f"{action_labels[a_i]}\n"
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
