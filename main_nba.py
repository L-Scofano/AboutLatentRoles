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


# Model
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

    # if args.model == "frozen":
    #     model = Model(args.input_dim, 
    #                     args.obs, 
    #                     args.preds, 
    #                     args.st_gcnn_dropout, 
    #                     args.entities,         
    #                     n_txcnn_layers=args.n_tcnn_layers,
    #                     txc_kernel_size=args.tcnn_kernel_size,
    #                     txc_dropout=args.tcnn_dropout,
    #                     reg = 0.001,
    #                     scale = 0.001,
    #                     pretrained=True).to(device)

    if args.model == "frozen":
        model = ModelVanilla(args.input_dim, 
                        args.obs, 
                        args.preds, 
                        args.st_gcnn_dropout, 
                        args.entities,         
                        n_txcnn_layers=args.n_tcnn_layers,
                        txc_kernel_size=args.tcnn_kernel_size,
                        txc_dropout=args.tcnn_dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    print("Total number of parameters of the network:", sum(p.numel() for p in model.parameters() if p.requires_grad))

init_model()

if args.model == "frozen":
    euc = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)
    euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
    order_nn_pre = SorterishWrapper(euc, args.reg, args.scale).to(device)
    # euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
    
if args.initialize:
    euc = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)
    order_nn_pre = SorterishWrapper(euc, args.reg, args.scale).to(device)
    # euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
    order_nn_pre.load_state_dict(torch.load(args.model_path+"/order_nn/best_order_nn_"+str(args.reg)+"_"+str(args.lr)+".pth"))
    print("Loaded model")


criterion = nn.MSELoss()

def train_e2e(verbose=True):

    if args.initialize:
        # initialize weights of ordernet
        model.ordernet.load_state_dict(order_nn_pre.state_dict())

        # model.ordernet.weight.data.copy_(order_nn_pre.weight.data)

    print('Loading training dataset')
    train_loss = []
    dataset_train = nba_dataset.Dataset(dataset_path+"train.npy", fields, args.obs+args.preds)
    if verbose: print(f"Training dataset size: {len(dataset_train)}")
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print('Loading validation dataset')
    val_loss = []
    dataset_val = nba_dataset.Dataset(dataset_path+"val.npy", fields, args.obs+args.preds)
    if verbose: print(f"Validation dataset size: {len(dataset_val)}")
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    best_val = [1e10, None]
    sw_gt = SorterishWrapper_GT()
    epochs = tqdm(range(args.n_epochs-1))


    for epoch in epochs:
        running_loss = 0
        n = 0
        model.train()
        for cnt, batch in enumerate(dataloader_train): 
            batch = batch.to(device)

            optimizer.zero_grad() 

            # INPUT 
            sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)
            sequences_train_to_rank = sequences_train[:,4:5,:,:]
            sequences_train = sequences_train.permute(0,3,1,2)


            # Re-order Input
            output, sorted_distance, og_distance, rank = get_output(sequences_train_to_rank)

            # Ground Truth
            sequences_gt = batch[:, args.obs:args.obs+args.preds].view(-1, args.preds, batch.shape[2]//2, 2)
            sequences_gt = rank_to_coordinates(sequences_gt, rank)
            sequences_gt = torch.from_numpy(np.array(sequences_gt)).to(device)

            sequences_predict, rank_predict, scores = model(sequences_train)
            sequences_predict = sequences_predict.permute(0,1,3,2)

            acc = 0
            for i in range (0, sequences_train.shape[0]):
                acc += accuracy_score(rank[i].astype(int), rank_predict[i].cpu().detach().numpy().astype(int))

            acc = acc/sequences_train.shape[0]
            # # transform array to tensor
            # rank = torch.from_numpy(rank).float().to(device)
            # sequences_gt = sw_gt(sequences_gt, rank)

            if args.supervised:
                loss_prediction = MAD(sequences_predict,sequences_gt)
                og_distance = torch.from_numpy(np.array(og_distance)).to(device)
                loss_euc = criterion(scores, og_distance)
                loss = args.alpha*loss_prediction + (1-args.alpha)*loss_euc

            
            if args.supervised == False:
                loss = MAD(sequences_predict,sequences_gt)

            if args.wandb:
                wandb.log({"loss": loss})


            if args.wandb:
                wandb.log({"accuracy": acc})

            epochs.set_description(f"[{cnt+1}/{len(dataloader_train)}] training loss: {loss.item()}")
    
            loss.backward()  
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)

            optimizer.step()
            running_loss += loss*batch.shape[0]
            n += batch.shape[0]
        train_loss.append(running_loss/n)  
        
        model.eval()
        with torch.no_grad():
            running_loss = 0 
            n = 0
            for cnt,batch in enumerate(dataloader_val):
                batch = batch.to(device)

                 # INPUT 
                sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)
                sequences_train_to_rank = sequences_train[:,4:5,:,:]
                sequences_train = sequences_train.permute(0,3,1,2)


                # Re-order Input
                output, sorted_distance, og_distance, rank = get_output(sequences_train_to_rank)

                # Ground Truth
                sequences_gt = batch[:, args.obs:args.obs+args.preds].view(-1, args.preds, batch.shape[2]//2, 2)
                sequences_gt = rank_to_coordinates(sequences_gt, rank)
                sequences_gt = torch.from_numpy(np.array(sequences_gt)).to(device)

                sequences_predict, rank_predict, scores = model(sequences_train)
                sequences_predict = sequences_predict.permute(0,1,3,2)

                acc = 0
                for i in range (0, sequences_train.shape[0]):
                    acc += accuracy_score(rank[i].astype(int), rank_predict[i].cpu().detach().numpy().astype(int))

                acc = acc/sequences_train.shape[0]

                # transform array to tensor
                # rank = torch.from_numpy(rank).float().to(device)
                # sequences_gt = sw_gt(sequences_gt, rank)

                if args.supervised:
                    loss_prediction = MAD(sequences_predict,sequences_gt)
                    og_distance = torch.from_numpy(np.array(og_distance)).to(device)
                    loss_euc = criterion(scores, og_distance)
                    loss = args.alpha*loss_prediction + (1-args.alpha)*loss_euc

                
                if args.supervised == False:
                    loss = MAD(sequences_predict,sequences_gt)


                if args.wandb:
                    wandb.log({"val loss": loss})


                if args.wandb:
                    wandb.log({"val accuracy": acc})

                epochs.set_description(f"[{cnt+1}/{len(dataloader_train)}] val loss: {loss.item()}")

                running_loss += loss*batch.shape[0]
                n += batch.shape[0]
            val_loss.append(running_loss/n)
            
        if args.use_scheduler:
            scheduler.step()

        if (val_loss[-1] < best_val[0]):
            best_val = [val_loss[-1], epoch]
            if args.save:
                torch.save(model.state_dict(),args.model_path+"/e2e_"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr))
            print("new best val:", best_val[0])
    
    print("best val:", best_val)  



def train_frozen(verbose=True):

    # Regular Dataset
    print('Loading training dataset')
    train_loss = []
    dataset_train = nba_dataset.Dataset(dataset_path+"train.npy", fields, args.obs+args.preds)
    if verbose: print(f"Training dataset size: {len(dataset_train)}")
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print('Loading validation dataset')
    val_loss = []
    dataset_val = nba_dataset.Dataset(dataset_path+"val.npy", fields, args.obs+args.preds)
    if verbose: print(f"Validation dataset size: {len(dataset_val)}")
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    best_val = [1e10, None]

    sw_gt = SorterishWrapper_GT()
    epochs = tqdm(range(args.n_epochs-1))


    for epoch in epochs:
        running_loss = 0
        n = 0
        model.train()
        for cnt, batch in enumerate(dataloader_train): 
            batch = batch.to(device)
            optimizer.zero_grad() 


            if args.gt:
                sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)
                sequences_train_to_rank = sequences_train[:,4:5,:,:]
                sequences_gt = batch[:, args.obs:args.obs+args.preds].view(-1, args.preds, batch.shape[2]//2, 2)
                
                # Create GT based on euclidean distance ranking
                input = sequences_train_to_rank.cpu().detach().numpy()
                sequence_gt, sorted_distance, og_distance, rank = get_output(input)

                # Order GT and Training set based on ranking
                rank = torch.from_numpy(rank).float().to(device)
                sequences_gt = sw_gt(sequences_gt, rank)
                sequences_train = sw_gt(sequences_train, rank)

                sequences_predict = model(sequences_train)
                sequences_predict = sequences_predict.permute(0,1,3,2)

                loss = MAD(sequences_predict,sequences_gt)
            
            else:
                
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
                acc = 0
                for i in range (0, sequences_train.shape[0]):
                    acc += accuracy_score(rank[i].astype(int), rank_predict[i].cpu().detach().numpy().astype(int))

                acc = acc/sequences_train.shape[0]

                if args.wandb:
                    wandb.log({"accuacy": acc})

                sequences_predict = sequences_predict.permute(0,1,3,2)
                rank = torch.from_numpy(rank).to(device)

                loss = MAD(sequences_predict,sequences_gt)


            if args.wandb:
                wandb.log({"loss": loss})

            epochs.set_description(f"[{cnt+1}/{len(dataloader_train)}] training loss: {loss.item()}")
    
            loss.backward()  
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)

            optimizer.step()
            running_loss += loss*batch.shape[0]
            n += batch.shape[0]
        train_loss.append(running_loss/n)  
        
        model.eval()
        with torch.no_grad():
            running_loss = 0 
            n = 0
            for cnt,batch in enumerate(dataloader_val):
                batch = batch.to(device)


                if args.gt:
                    sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)
                    
                    sequences_train_to_rank = sequences_train[:,4:5,:,:]
                    sequences_gt = batch[:, args.obs:args.obs+args.preds].view(-1, args.preds, batch.shape[2]//2, 2)
                    
                    # Create GT based on euclidean distance ranking
                    input = sequences_train_to_rank.cpu().detach().numpy()
                    sequence_gt, sorted_distance, og_distance, rank = get_output(input)

                    # Order GT and Training set based on ranking
                    rank = torch.from_numpy(rank).float().to(device)
                    sequences_gt = sw_gt(sequences_gt, rank)
                    sequences_train = sw_gt(sequences_train, rank)
                    # sequences_train = sequences_train.permute(0,3,1,2)


                    sequences_predict = model(sequences_train)
                    sequences_predict = sequences_predict.permute(0,1,3,2)

                    # # convert tensor to array 
                    # rank = rank.cpu().detach().numpy()
                    # acc = 0
                    # for i in range (0, sequences_train.shape[0]):
                    #     acc += accuracy_score(rank[i].astype(int), rank_predict[i].cpu().detach().numpy().astype(int))

                    # acc = acc/sequences_train.shape[0]

                    loss = MAD(sequences_predict,sequences_gt)
            
                else:
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
                    acc = 0
                    for i in range (0, sequences_train.shape[0]):
                        acc += accuracy_score(rank[i].astype(int), rank_predict[i].cpu().detach().numpy().astype(int))

                    acc = acc/sequences_train.shape[0]

                    if args.wandb:
                        wandb.log({"accuacy": acc})

                    sequences_predict = sequences_predict.permute(0,1,3,2)
                    rank = torch.from_numpy(rank).to(device)

                    loss = MAD(sequences_predict,sequences_gt)


                if args.wandb:
                    wandb.log({"val loss": loss})

                epochs.set_description(f"[{cnt+1}/{len(dataloader_train)}] val loss: {loss.item()}")

                running_loss += loss*batch.shape[0]
                n += batch.shape[0]
            val_loss.append(running_loss/n)
            
        if args.use_scheduler:
            scheduler.step()

        if (val_loss[-1] < best_val[0]):
            best_val = [val_loss[-1], epoch]
            if args.save:
                torch.save(model.state_dict(),args.model_path+"/e2e_"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr))
            print("new best val:", best_val[0])
    
    print("best val:", best_val)  


    '''
    # Oracle coordinates
    train_rank_oracle = torch.load("Oracle Dataset/train_ranking_oracle.npy")
    train_rank_oracle = torch.from_numpy(np.array(train_rank_oracle))
    train_rank_oracle = torch.split(train_rank_oracle, args.batch_size) 

    test_rank_oracle = torch.load("Oracle Dataset/test_ranking_oracle.npy")
    test_rank_oracle = torch.from_numpy(np.array(test_rank_oracle))
    test_rank_oracle = torch.split(test_rank_oracle, args.batch_size)

    val_rank_oracle = torch.load("Oracle Dataset/val_ ranking_oracle.npy")
    val_rank_oracle = torch.from_numpy(np.array(val_rank_oracle))
    val_rank_oracle = torch.split(val_rank_oracle, args.batch_size)'''



# main function
if __name__ == "__main__":


    if args.wandb:
        print(args.model, 'players = '+str(args.entities), "batch_size ="+str(args.batch_size), "lr ="+str(args.lr), "mode ="+args.mode)
        wandb.login()
        wandb.init(
            project="NBA",
            entity="lscofano",
            tags=[args.model, 'players = '+str(args.entities), "batch_size ="+str(args.batch_size), "lr ="+str(args.lr), "mode ="+args.mode],
            config=vars(args),
        )


    if args.model == "e2e":
        if args.mode == "train":
            print("End to end model")
            train_e2e()  

    if args.model == "frozen":
        if args.mode == "train":
            print("Frozen model")
            train_frozen()  

    # if args.order_nn:
    #     if args.mode == "train":
    #         train_order_nn()
    #         print("Training Order NN")
    #     elif args.mode == "test":
    #         test_order_nn()
    #         print("Testing Order NN")


    if args.wandb:
        wandb.finish()







