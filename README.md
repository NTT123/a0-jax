# a0-jax
AlphaZero in JAX using PAX library.

```sh
pip install -r requirements.txt
```


## Train agent

### Connect-Two game


```sh
python train_agent.py
```


### Connect-Four game

```sh
TF_CPP_MIN_LOG_LEVEL=2 \
python train_agent.py \
    --game_class="connect_four_game.Connect4Game" \
    --agent_class="resnet_policy.ResnetPolicyValueNet" \
    --batch-size=4096 \
    --num_simulations_per_move=32 \
    --num_self_plays_per_iteration=102400 \
    --learning-rate=1e-2 \
    --num_iterations=500 \
    --lr-decay-steps=200000
```

A trained Connect-4 agent is running at https://huggingface.co/spaces/ntt123/Connect-4-Game. We use tensorflow.js to run the policy on the browser.


### Go game

```sh
TF_CPP_MIN_LOG_LEVEL=2 \
python3 train_agent.py \
    --game-class="go_game.GoBoard9x9" \
    --agent-class="resnet_policy.ResnetPolicyValueNet256" \
    --selfplay-batch-size=1024 \
    --training-batch-size=1024 \
    --num-simulations-per-move=32 \
    --num-self-plays-per-iteration=102400 \
    --learning-rate=1e-3 \
    --random-seed=42 \
    --ckpt-filename="./go_agent_9x9_256.ckpt" \
    --num-iterations=200 \
    --lr-decay-steps=200000
```

A live Go agent is running at https://go.ntt123.repl.co.
You can run the agent on your local machine with the `go_web_app.py` script.

We also have an [interative colab notebook](https://colab.research.google.com/drive/1IlN1gThYrLazxTGrhryNzspx-Ts_6llj?usp=sharing) that runs the agent on GPU to reduce inference time.


## Plot the search tree

```sh
python plot_search_tree.py 
# ./search_tree.png
```

## Play

```sh
python play.py
```
