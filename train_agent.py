"""
AlphaZero training script.

Train agent by self-play only.
"""

import pickle
import random
from functools import partial

import chex
import click
import fire
import jax
import jax.numpy as jnp
import numpy as np
import opax
import optax
import pax

from env import Enviroment
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


@partial(jax.jit, static_argnames=("batch_size", "num_simulations_per_move"))
def collect_batched_self_play_data(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    *,
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
        state = env.canonical_observation()
        terminated = env.is_terminated()
        policy_output = improve_policy_with_mcts(
            agent,
            env,
            recurrent_fn,
            rng_key,
            num_simulations_per_move,
            temperature=jnp.where(step > 20, 0.3, 1.0),
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


def collect_self_play_data(
    agent,
    env,
    rng_key: chex.Array,
    batch_size: int,
    data_size: int,
    num_simulations_per_move: int,
):
    """Collect self-play data for training."""
    N = data_size // batch_size
    rng_keys = jax.random.split(rng_key, N)
    data = []

    with click.progressbar(rng_keys, label="  self play  ") as bar:
        for rng_key in bar:
            batch = collect_batched_self_play_data(
                agent,
                env,
                rng_key,
                batch_size=batch_size,
                num_simulations_per_move=num_simulations_per_move,
            )
            data.append(jax.device_get(batch))
    data = jax.tree_map(lambda *xs: np.concatenate(xs), *data)
    return data


def prepare_training_data(data: MoveOutput):
    """Preprocess the data collected from self-play.

    1. remove states after the enviroment is terminated.
    2. compute the value at each state.
    """
    buffer = []
    N = len(data.terminated)
    for i in range(N):
        state = data.state[i]
        is_terminated = data.terminated[i]
        action_weights = data.action_weights[i]
        reward = data.reward[i]
        L = len(is_terminated)
        value = None
        for idx in reversed(range(L)):
            if is_terminated[idx]:
                continue
            value = reward[idx] if value is None else -value
            buffer.append(
                TrainingExample(
                    state=state[idx],
                    action_weights=action_weights[idx],
                    value=np.array(value, dtype=np.float32),
                )
            )

    return buffer


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


@jax.jit
def train_step(net, optim, data: TrainingExample):
    """A training step."""
    (_, (net, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(net, data)
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses


def train(
    game_class="connect_two_game.Connect2Game",
    agent_class="mlp_policy.MlpPolicyValueNet",
    batch_size: int = 32,
    num_iterations: int = 50,
    num_simulations_per_move: int = 16,
    num_self_plays_per_iteration: int = 1024,
    learing_rate: float = 0.001,
    ckpt_filename: str = "./agent.ckpt",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
):
    """Train an agent by self-play."""
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )
    rng_key = jax.random.PRNGKey(random_seed)
    shuffler = random.Random(random_seed)

    optim = opax.adamw(learing_rate, weight_decay=weight_decay).init(agent.parameters())

    for iteration in range(num_iterations):
        print(f"Iteration {iteration}")
        rng_key_1, rng_key = jax.random.split(rng_key, 2)
        agent = agent.eval()
        data = collect_self_play_data(
            agent,
            env,
            rng_key_1,
            batch_size,
            num_self_plays_per_iteration,
            num_simulations_per_move,
        )
        buffer = prepare_training_data(data)
        shuffler.shuffle(buffer)
        buffer = jax.tree_map(lambda *xs: np.stack(xs), *buffer)
        N = buffer.state.shape[0]
        losses = []
        agent = agent.train()
        with click.progressbar(
            range(0, N - batch_size, batch_size), label="  train agent"
        ) as bar:
            for i in bar:
                batch = jax.tree_map(lambda x: x[i : (i + batch_size)], buffer)
                agent, optim, loss = train_step(agent, optim, batch)
                losses.append(loss)

        value_loss, policy_loss = zip(*losses)
        value_loss = sum(value_loss).item() / len(value_loss)
        policy_loss = sum(policy_loss).item() / len(policy_loss)
        print(f"  train losses:  value {value_loss:.3f}  policy {policy_loss:.3f}")
        # save agent's weights to disk
        with open(ckpt_filename, "wb") as f:
            pickle.dump(agent.state_dict(), f)
    print("Done!")


if __name__ == "__main__":
    fire.Fire(train)
