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
--agent_class="resnet_policy_net.ResnetPolicyValueNet" \
--batch-size=512 \
--num_simulations_per_move=512 \
--num_self_plays_per_iteration=2048 \
--learning-rate=1e-4
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