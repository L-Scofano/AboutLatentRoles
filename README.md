<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

The official PyTorch implementation of the **AI4ABM ICLR '23 Workshop** paper [About latent roles in forecasting players in team sports
](https://arxiv.org/abs/2304.08272).

Visit our [webpage](https://www.pinlab.org/aboutlatentroles) for more details.

![teaser](teaser.png)

## Setup
To install the required packages, run the following command:

```
conda create -n latent_roles python=3.6
conda activate latent_roles
pip install -r requirements.txt
```
As we point out in the paper there are several components in the network, and all are trainable separately. We provide the following scripts to train and test the different components.

## Order Neural Network

OrderNN is a module that allows the permutation of the players such that each player is assigned to a latent role. 

### Train Euclidean Distance
```
python main_nba_order_nn.py --euclidean_distance --model euclidean  --n_epoch 50 --wandb
```

### Train Order NN
```
python main_nba.py --order_nn --model order_nn --n_epoch 50  --wandb
python main_nba.py --order_nn --model order_nn --n_epoch 50 --obs 1 --wandb
```
## Test
-> We have two Ground Truth orderings:
- ordering 1 = getoutput / SorterishWrapperTest
- ordeing 2 = reordering_dataset / SorterishWrapperTest2

### Test Order NN
```
python main_nba_order_nn.py --order_nn --model order_nn --mode test --ordering1
```

# E2E
