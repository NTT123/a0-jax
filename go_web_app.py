"""go game web server"""

import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict

import jax
import jax.numpy as jnp
from flask import Flask, jsonify, redirect, request, send_file, url_for

from play import play_one_move
from utils import env_step, import_class, reset_env

parser = ArgumentParser()
parser.add_argument("--game-class", default="go_game.GoBoard9x9", type=str)
parser.add_argument(
    "--agent-class", default="resnet_policy.ResnetPolicyValueNet256", type=str
)
parser.add_argument("--ckpt-filename", default="go_agent_9x9_256.ckpt", type=str)
parser.add_argument("--num_simulations_per_move", default=1024, type=int)
disable_mcts = False
args = parser.parse_args()

ENV = import_class(args.game_class)()
AGENT = import_class(args.agent_class)(
    input_dims=ENV.observation().shape,
    num_actions=ENV.num_actions(),
)

with open(args.ckpt_filename, "rb") as f:
    AGENT = AGENT.load_state_dict(pickle.load(f)["agent"])
AGENT = AGENT.eval()

all_games: Dict[int, Any] = defaultdict(import_class(args.game_class))


def human_vs_agent(env, info):
    """A game of human vs agent."""
    human_action = info["human_action"]
    if human_action == -1:
        # Agent goes first
        env = reset_env(env)
        env.render()
    else:
        if human_action == "pass":
            human_action = env.num_actions() - 1
        action = jnp.array(human_action, dtype=jnp.int32)
        env, reward = env_step(env, action)
        if env.is_terminated().item():
            reward = reward.item()
            if reward == 1:
                msg = "You won!"
            elif reward == -1:
                msg = "You lost :-("
            else:
                msg = ""

            return env, {
                "action": -1,
                "terminated": env.is_terminated().item(),
                "current_board": env.board.reshape((-1,)).tolist(),
                "msg": msg,
            }
    rng_key = jax.random.PRNGKey(random.randint(0, 999999))
    action, action_weights, value = play_one_move(
        AGENT,
        env,
        rng_key,
        disable_mcts=disable_mcts,
        num_simulations=args.num_simulations_per_move,
        random_action=False,
    )
    del action_weights, value
    env, reward = env_step(env, action)
    reward = reward.item()
    if reward == -1:
        msg = "You won!"
    elif reward == 1:
        msg = "You lost :-("
    else:
        msg = ""
    action = action.item()
    if len(msg) == 0 and action == env.num_actions() - 1:
        msg = "AI PASSED!"
    return env, {
        "action": action,
        "terminated": env.is_terminated().item(),
        "current_board": env.board.reshape((-1,)).tolist(),
        "msg": msg,
    }


app = Flask(__name__)


@app.route("/<int:gameid>/move", methods=["POST"])
def move(gameid: int):
    env = all_games[gameid]
    info = request.get_json()
    env, res = human_vs_agent(env, info)
    all_games[gameid] = env
    return jsonify(res)


@app.route("/<int:gameid>", methods=["GET"])
def startgame(gameid: int):
    all_games[gameid] = reset_env(all_games[gameid])
    return send_file("./index.html")


@app.route("/")
def index():
    env = import_class(args.game_class)()
    gameid = random.randint(0, 999999)
    all_games[gameid] = env
    return redirect(url_for("startgame", gameid=gameid))


@app.route("/<int:gameid>/reset")
def reset(gameid: int):
    all_games[gameid] = reset_env(all_games[gameid])
    return {}


@app.route("/stone.ogg")
def stone():
    return send_file("./stone.ogg")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
