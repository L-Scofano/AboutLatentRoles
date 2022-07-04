from ast import arg
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.autograd
import torch.nn as nn
import torch
import numpy as np

import wandb

from tqdm.notebook import tqdm
from utils.parser import args

from utils.loss_funcs import *
from utils import nba_dataset

# To reload stuff
import importlib
importlib.reload(nba_dataset)

import utils.loss_funcs
importlib.reload(utils.loss_funcs)
from utils.loss_funcs import *
from utils_func import *

from models.order_nn import *

from models.sts_gcn import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
datasets_base = "datasets/"
dataset_name = "nba_50k_2.5fps_25frames_seed100_split2_50.100pctl"
fields = nba_dataset.TEAM_HOME + nba_dataset.TEAM_AWAY + nba_dataset.BALL_XY



#  Model
tag = "bestmodel"

dataset_path = f"{datasets_base}/{dataset_name}/"
    
  
# Training
clip_grad=None # select max norm to clip gradients

device, model, optimizer, scheduler = [None]*4
def init_model():
    global device, model, optimizer, scheduler
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s'%device)

    if args.model == "e2e":
        model = Model(args.input_dim, 
                            args.obs, 
                            args.preds, 
                            args.st_gcnn_dropout, 
                            args.entities,         
                            n_txcnn_layers=args.n_tcnn_layers,
                            txc_kernel_size=args.tcnn_kernel_size,
                            txc_dropout=args.tcnn_dropout,
                            reg = 0.001,
                            scale = 0.001).to(device)
        # load weight for model
        print('Here1')
        model.load_state_dict(torch.load(args.model_path+"/e2e_"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))

    elif args.model == "frozen":
        model = ModelVanilla(args.input_dim, 
                        args.obs, 
                        args.preds, 
                        args.st_gcnn_dropout, 
                        args.entities,         
                        n_txcnn_layers=args.n_tcnn_layers,
                        txc_kernel_size=args.tcnn_kernel_size,
                        txc_dropout=args.tcnn_dropout).to(device)
        # load weight for model
        print('Here2')

        model.load_state_dict(torch.load(args.model_path+"/e2e_"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))

    print("Total number of parameters of the network:", sum(p.numel() for p in model.parameters() if p.requires_grad))

init_model()


if args.model == "frozen":
    euc = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)
    euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
    order_nn_pre = SorterishWrapper(euc, args.reg, args.scale).to(device)
    # euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
    

def test(verbose=True):
    # if path is None: path = checkpoint_dir+model_name+"_best"
    # model.load_state_dict(torch.load(path))
    model.eval()
    accum_loss = [0,0]
    
    n_batches = 0 # number of batches for all the sequences
    print('Loading test dataset')
    dataset_test = nba_dataset.Dataset(dataset_path+"test.npy", fields, args.obs+args.preds)
    dataset_test = ordering_data(dataset_test)
    #dataset_test = dataset_reordering(dataset_test, 'away_minmatch_mixed')
    if verbose: print(f"Test dataset size: {len(dataset_test)}")
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)

    for cnt, batch in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        with torch.no_grad():
            batch = batch.to(device)
            n_batches += batch.shape[0]



            if args.model == "frozen":
                
                # Input
                sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)
                sequences_train_to_rank = sequences_train[:,4:5,:,:].contiguous()
                sequences_train_not_to_rank = sequences_train[:,:4,:,:].contiguous()

                # --> Pass into Order NN to produce permuted coordinates

                # Input
                sequences_train, rank_predict, scores_train = order_nn_pre(sequences_train_not_to_rank, sequences_train_to_rank)
                sequences_train = sequences_train.permute(0,3,1,2)
                input = sequences_train_to_rank.cpu().detach().numpy()
                output, sorted_distance, og_distance, rank = get_output(input)

                # Order GT and Training set based on ranking
                sequences_gt = batch[:, args.obs:args.obs+args.preds].view(-1, args.preds, batch.shape[2]//2, 2)
                sequences_gt = rank_to_coordinates(sequences_gt, rank)
                sequences_gt = torch.from_numpy(np.array(sequences_gt)).to(device)


                sequences_predict = model(sequences_train)

            if args.model == "e2e":

                sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2).permute(0,3,1,2)
                sequences_gt = batch[:, args.obs:args.obs+args.preds]

                sequences_predict, rank, scores= model(sequences_train)


            sequences_predict = sequences_predict.permute(0,1,3,2).contiguous().view(-1, args.preds, batch.shape[2])
            
            accum_loss[0] += MAD(sequences_predict, sequences_gt) * batch.shape[0]
            accum_loss[1] += FAD(sequences_predict, sequences_gt) * batch.shape[0]
    print(f'MAD: {accum_loss[0]/n_batches:.3f} / FAD: {accum_loss[1]/n_batches:.3f}')



# main
if __name__ == "__main__":
    # test()
    test()