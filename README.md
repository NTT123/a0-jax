# a0-jax
AlphaZero in JAX using PAX library.

```sh
pip install -r requirements.txt
```


## Train agent

#### Connect-Two game


```sh
python train_agent.py
```


#### Connect-Four game

```sh
TF_CPP_MIN_LOG_LEVEL=2 \
python train_agent.py \
--game_class="connect_four_game.Connect4Game" \
--agent_class="resnet_policy.ResnetPolicyValueNet" \
--batch-size=4096 \
--num_simulations_per_move=32 \
--num_self_plays_per_iteration=16384 \
--learning-rate=1e-4 \
--buffer-size=2000000 \
--num_iterations=1000
```

A trained Connect-4 agent is running at https://huggingface.co/spaces/ntt123/Connect-4-Game. We use tensorflow.js to run the policy on the browser.


## Plot the search tree

```sh
python plot_search_tree.py 
# ./search_tree.png
```

## Play

```sh
python play.py
```
