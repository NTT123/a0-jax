"""
AlphaZero training script.

Train agent by self-play only.
"""

import os
import pickle
import random
from functools import partial
from typing import Optional

import chex
import click
import fire
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import numpy as np
import opax
import optax
import pax

from games.env import Enviroment
from play import PlayResults, agent_vs_agent_multiple_games
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import batched_policy, env_step, import_class, replicate, reset_env

EPSILON = 1e-9  # a very small positive value


@chex.dataclass(frozen=True)
class TrainingExample:
    """AlphaZero training example.

    state: the current state of the game.
    action_weights: the target action probabilities from MCTS policy.
    value: the target value from self-play result.
    """

    state: chex.Array
    action_weights: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class MoveOutput:
    """The output of a single self-play move.

    state: the current state of game.
    reward: the reward after execute the action from MCTS policy.
    terminated: the current state is a terminated state (bad state).
    action_weights: the action probabilities from MCTS policy.
    """

    state: chex.Array
    reward: chex.Array
    terminated: chex.Array
    action_weights: chex.Array


@partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=(3, 4))
def collect_batched_self_play_data(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    batch_size: int,
    num_simulations_per_move: int,
):
    """Collect a batch of self-play data using mcts."""

    def single_move(prev, inputs):
        """Execute one self-play move using MCTS.

        This function is designed to be compatible with jax.scan.
        """
        env, rng_key, step = prev
        del inputs
        rng_key, rng_key_next = jax.random.split(rng_key, 2)
        state = jax.vmap(lambda e: e.canonical_observation())(env)
        terminated = env.is_terminated()
        policy_output = improve_policy_with_mcts(
            agent,
            env,
            rng_key,
            recurrent_fn,
            num_simulations_per_move,
        )
        env, reward = jax.vmap(env_step)(env, policy_output.action)
        return (env, rng_key_next, step + 1), MoveOutput(
            state=state,
            action_weights=policy_output.action_weights,
            reward=reward,
            terminated=terminated,
        )

    env = reset_env(env)
    env = replicate(env, batch_size)
    step = jnp.array(1)
    _, self_play_data = pax.scan(
        single_move,
        (env, rng_key, step),
        None,
        length=env.max_num_steps(),
        time_major=False,
    )
    return self_play_data


def prepare_training_data(data: MoveOutput, env: Enviroment):
    """Preprocess the data collected from self-play.

    1. remove states after the enviroment is terminated.
    2. compute the value at each state.
    """
    buffer = []
    num_games = len(data.terminated)
    for i in range(num_games):
        state = data.state[i]
        is_terminated = data.terminated[i]
        action_weights = data.action_weights[i]
        reward = data.reward[i]
        num_steps = len(is_terminated)
        value: Optional[chex.Array] = None
        for idx in reversed(range(num_steps)):
            if is_terminated[idx]:
                continue
            if value is None:
                value = reward[idx]
            else:
                value = -value
            s = np.copy(state[idx])
            a = np.copy(action_weights[idx])
            for augmented_s, augmented_a in env.symmetries(s, a):
                buffer.append(
                    TrainingExample(  # type: ignore
                        state=augmented_s,
                        action_weights=augmented_a,
                        value=np.array(value, dtype=np.float32),
                    )
                )

    return buffer


def collect_self_play_data(
    agent,
    env,
    rng_key: chex.Array,
    batch_size: int,
    data_size: int,
    num_simulations_per_move: int,
):
    """Collect self-play data for training."""
    num_iters = data_size // batch_size
    devices = jax.local_devices()
    num_devices = len(devices)
    rng_key_list = jax.random.split(rng_key, num_iters * num_devices)
    rng_keys = jnp.stack(rng_key_list).reshape((num_iters, num_devices, -1))  # type: ignore
    data = []

    with click.progressbar(range(num_iters), label="  self play     ") as bar:
        for i in bar:
            batch = collect_batched_self_play_data(
                agent,
                env,
                rng_keys[i],
                batch_size // num_devices,
                num_simulations_per_move,
            )
            batch = jax.device_get(batch)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, *x.shape[2:])), batch
            )
            data.extend(prepare_training_data(batch, env=env))
    return data


def loss_fn(net, data: TrainingExample):
    """Sum of value loss and policy loss."""
    net, (action_logits, value) = batched_policy(net, data.state)

    # value loss (mse)
    mse_loss = optax.l2_loss(value, data.value)
    mse_loss = jnp.mean(mse_loss)

    # policy loss (KL(target_policy', agent_policy))
    target_pr = data.action_weights
    # to avoid log(0) = nan
    target_pr = jnp.where(target_pr == 0, EPSILON, target_pr)
    action_logits = jax.nn.log_softmax(action_logits, axis=-1)
    kl_loss = jnp.sum(target_pr * (jnp.log(target_pr) - action_logits), axis=-1)
    kl_loss = jnp.mean(kl_loss)

    # return the total loss
    return mse_loss + kl_loss, (net, (mse_loss, kl_loss))


@partial(jax.pmap, axis_name="i")
def train_step(net, optim, data: TrainingExample):
    """A training step."""
    (_, (net, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(net, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses


def train(
    game_class="games.connect_two_game.Connect2Game",
    agent_class="policies.mlp_policy.MlpPolicyValueNet",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,
    num_iterations: int = 100,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 0.01,
    ckpt_filename: str = "./agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
):
    """Train an agent by self-play."""
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )

    def lr_schedule(step):
        e = jnp.floor(step * 1.0 / lr_decay_steps)
        return learning_rate * jnp.exp2(-e)

    optim = opax.chain(
        opax.add_decayed_weights(weight_decay),
        opax.sgd(lr_schedule, momentum=0.9),
    ).init(agent.parameters())

    if os.path.isfile(ckpt_filename):
        print("Loading weights at", ckpt_filename)
        with open(ckpt_filename, "rb") as f:
            dic = pickle.load(f)
            agent = agent.load_state_dict(dic["agent"])
            optim = optim.load_state_dict(dic["optim"])
            start_iter = dic["iter"] + 1
    else:
        start_iter = 0
    rng_key = jax.random.PRNGKey(random_seed)
    shuffler = random.Random(random_seed)
    devices = jax.local_devices()
    num_devices = jax.local_device_count()

    def _stack_and_reshape(*xs):
        x = np.stack(xs)
        x = np.reshape(x, (num_devices, -1) + x.shape[1:])
        return x

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")
        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)
        agent = agent.eval()
        data = collect_self_play_data(
            agent,
            env,
            rng_key_1,  # type: ignore
            selfplay_batch_size,
            num_self_plays_per_iteration,
            num_simulations_per_move,
        )
        data = list(data)
        shuffler.shuffle(data)
        old_agent = jax.tree_util.tree_map(jnp.copy, agent)
        agent, losses = agent.train(), []
        agent, optim = jax.device_put_replicated((agent, optim), devices)
        ids = range(0, len(data) - training_batch_size, training_batch_size)
        with click.progressbar(ids, label="  train agent   ") as progressbar:
            for idx in progressbar:
                batch = data[idx : (idx + training_batch_size)]
                batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
                agent, optim, loss = train_step(agent, optim, batch)
                losses.append(loss)

        value_loss, policy_loss = zip(*losses)
        value_loss = np.mean(sum(jax.device_get(value_loss))) / len(value_loss)
        policy_loss = np.mean(sum(jax.device_get(policy_loss))) / len(policy_loss)
        agent, optim = jax.tree_util.tree_map(lambda x: x[0], (agent, optim))
        # new agent is player 1
        result_1: PlayResults = agent_vs_agent_multiple_games(
            agent.eval(),
            old_agent,
            env,
            rng_key_2,
            num_simulations_per_move=32,
        )
        # old agent is player 1
        result_2: PlayResults = agent_vs_agent_multiple_games(
            old_agent,
            agent.eval(),
            env,
            rng_key_3,
            num_simulations_per_move=32,
        )
        print(
            "  evaluation      {} win - {} draw - {} loss".format(
                result_1.win_count + result_2.loss_count,
                result_1.draw_count + result_2.draw_count,
                result_1.loss_count + result_2.win_count,
            )
        )
        print(
            f"  value loss {value_loss:.3f}"
            f"  policy loss {policy_loss:.3f}"
            f"  learning rate {optim[1][-1].learning_rate:.1e}"
        )
        # save agent's weights to disk
        with open(ckpt_filename, "wb") as writer:
            dic = {
                "agent": jax.device_get(agent.state_dict()),
                "optim": jax.device_get(optim.state_dict()),
                "iter": iteration,
            }
            pickle.dump(dic, writer)
    print("Done!")


if __name__ == "__main__":
    if "COLAB_TPU_ADDR" in os.environ:
        jax.tools.colab_tpu.setup_tpu()
    print("Cores:", jax.local_devices())

    fire.Fire(train)
