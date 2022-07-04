# Order NN
## Train
### Euclidean Distance
python main_nba_order_nn.py --euclidean_distance --model euclidean  --n_epoch 50 --wandb


### Order NN
python main_nba.py --order_nn --model order_nn --n_epoch 50  --wandb
python main_nba.py --order_nn --model order_nn --n_epoch 50 --obs 1 --wandb

## Test
-> We have two ordering GT:
- ordering 1 = getoutput / SorterishWrapperTest
- ordeing 2 = reordering_dataset / SorterishWrapperTest2

### Pre trained Order NN
python main_nba.py --order_nn --pre_trained --model order_nn --mode test --wandb 


# E2E
