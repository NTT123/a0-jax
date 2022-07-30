"""
Convert agent policy to tfjs model that runs on browser.

Usage:
    pip install tensorflowjs
    python convert_to_tfjs.py --agent_class="resnet_policy.ResnetPolicyValueNet" --game_class="connect_four_game.Connect4Game"
    tensorflowjs_converter --input_format=tf_saved_model --output_node_names='output_0,output_1' ./tf_saved_agent ./tf_saved_agent_js
"""

import pickle
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import tensorflow as tf
import tree
from fire import Fire
from jax.experimental import jax2tf

from utils import import_class


def create_variable(path, value):
    name = "/".join(map(str, path)).replace("~", "_")
    return tf.Variable(value, name=name)


def main(
    game_class: str = "connect_two_game.Connect2Game",
    agent_class="mlp_policy.MlpPolicyValueNet",
    ckpt_filename: str = "./agent.ckpt",
    tf_model_path: str = "./tf_saved_agent",
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

    inputs = (env.observation().astype(jnp.float32),)
    print(agent(inputs[0]))

    @partial(jax2tf.convert, with_gradient=True, enable_xla=False)
    def tf_forward(leaves, x):
        _, treedef = jax.tree_util.tree_flatten(agent)
        agent_ = jax.tree_util.tree_unflatten(treedef, leaves)
        y = agent_(x)
        return y

    agent = jax.device_put(agent)
    tf_params = tree.map_structure_with_path(create_variable, jax.tree_leaves(agent))

    @tf.function(autograph=False, input_signature=[tf.TensorSpec(inputs[0].shape)])
    def tfmodel_forward(x):
        return tf_forward(tf_params, x)

    model = tf.Module()
    model.f = tfmodel_forward
    model.params = tf_params
    o = model.f(*tree.map_structure(tf.zeros_like, inputs))  # Dummy call.
    print(o)
    tf.saved_model.save(model, tf_model_path)
    m1 = tf.saved_model.load(tf_model_path)
    o1 = m1.f(*tree.map_structure(tf.zeros_like, inputs))
    print(o1)

    cmd = f"tensorflowjs_converter --input_format=tf_saved_model --output_node_names='output_0,output_1' {tf_model_path} {tf_model_path}_js"
    print("Run the following command:")
    print(cmd)


if __name__ == "__main__":
    Fire(main)
