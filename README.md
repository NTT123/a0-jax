# a0-jax (work-in-progress)
AlphaZero in JAX using PAX library

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
python train_agent.py \
--game_class="connect_four_game.Connect4Game" \
--agent_class="resnet_policy.ResnetPolicyValueNet" \
--batch-size=512 \
--num_simulations_per_move=512 \
--num_self_plays_per_iteration=2048 \
--learning-rate=1e-4 \
--temperature-decay=0.95
```

#### Caro game

```sh
TF_CPP_MIN_LOG_LEVEL=2 \
python train_agent.py --game_class="caro_game.CaroGame" \
--agent_class="resnet_policy.ResnetPolicyValueNet" \
--batch-size=128 \
--num_simulations_per_move=512 \
--num_self_plays_per_iteration=4096 \
--learning-rate=1e-4 \
--temperature-decay=0.95
```

## Plot the search tree

```sh
python plot_search_tree.py 
# ./search_tree.png
```

## Play

```sh
python play.py
```