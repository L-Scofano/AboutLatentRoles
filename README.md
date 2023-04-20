<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

The official PyTorch implementation of the **AI4ABM ICLR '23 Workshop** paper [**About latent roles in forecasting players in team sports
**](https://arxiv.org/abs/2304.08272).

## Order NN

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
-> We have two ordering GT:
- ordering 1 = getoutput / SorterishWrapperTest
- ordeing 2 = reordering_dataset / SorterishWrapperTest2

### Test Order NN
```
python main_nba_order_nn.py --order_nn --model order_nn --mode test --ordering1
```

# E2E
