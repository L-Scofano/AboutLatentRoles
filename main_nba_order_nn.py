from sklearn.metrics import accuracy_score, euclidean_distances
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
from utils.loss_funcs import *
from utils.utils_func import *
from models.order_nn import *
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
    


if args.euclidean_distance:

    model = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)
    clip_grad=None # select max norm to clip gradients


    device, model, optimizer, scheduler = [None]*4
    def init_model():
        global device, model, optimizer, scheduler
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device: %s'%device)

        model = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

        if args.use_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        print("Total number of parameters of the network:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    init_model()


    criterion = nn.MSELoss()

    def train_euclidean(verbose=True):
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
        epochs = tqdm(range(args.n_epochs-1))


        for epoch in epochs:
            running_loss = 0
            n = 0
            model.train()
            for cnt, batch in enumerate(dataloader_train): 
                batch = batch.to(device)

                optimizer.zero_grad() 

                sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)

                sequences_train = sequences_train[:,4:5,:,:]
                # sequences_train_not_to_rank = sequences_train[:,:4,:,:] 
                # input = sequences_train_to_rank.cpu().detach().numpy()

                predicted_euc = model(sequences_train)

                input = sequences_train.cpu().detach().numpy()
                
                output, sorted_distance, og_distance, rank = get_output(input)
                og_distance = torch.from_numpy(np.array(og_distance)).to(device)

                
                # print(predicted_euc.shape, sequences_train_to_rank.shape)
                loss = criterion(og_distance,predicted_euc)

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

                
                    sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)

                    sequences_train = sequences_train[:,4:5,:,:]
                    # sequences_train_not_to_rank = sequences_train[:,:4,:,:] 
                    # input = sequences_train_to_rank.cpu().detach().numpy()
                    predicted_euc = model(sequences_train)


                    input = sequences_train.cpu().detach().numpy()
                    output, sorted_distance, og_distance, rank = get_output(input)
                    og_distance = torch.from_numpy(np.array(og_distance)).to(device)

                    loss = criterion(og_distance,predicted_euc)


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
                torch.save(model.state_dict(),args.model_path+"/euclidean/euc_best"+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr))
                print("new best val:", best_val[0])
        
        print("best val:", best_val)  




if args.order_nn:

    device, model, optimizer, scheduler = [None]*4
    clip_grad=None # select max norm to clip gradients


    def init_model():
        global device, model, optimizer, scheduler
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device: %s'%device)

        
        model = SorterishWrapper(SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim), args.reg, args.scale).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

        if args.use_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        print("Total number of parameters of the network:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    init_model()

    # %matplotlib notebook

    criterion = nn.MSELoss()

    def train_order_nn(verbose=True):
        print('Loading training dataset')
        train_loss = []
        dataset_train = nba_dataset.Dataset(dataset_path+"train.npy", fields, args.obs+args.preds)
        if verbose: print(f"Training dataset size: {len(dataset_train)}")
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

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

                sequences_train = batch[:, :args.obs].reshape(-1, args.obs, batch.shape[2]//2, 2)
                sequences_train_to_rank = sequences_train[:,4:5,:,:]
                sequences_train_not_to_rank = sequences_train[:,:4,:,:] 

                sorted_x, rank_pred, scores= model(sequences_train_not_to_rank,sequences_train_to_rank)

                input = sequences_train_to_rank.cpu().detach().numpy()
                sequence_gt, sorted_distance, og_distance, rank = get_output(input)

                # sequences_train_not_to_rank = rank_to_coordinates(sequences_train_not_to_rank, rank)
                # sequences_train_not_to_rank = torch.from_numpy(sequences_train_not_to_rank).to(device)
                # sequence_gt = torch.from_numpy(np.array(sequence_gt)).to(device)
                # sequences_train_gt = torch.cat((sequences_train_not_to_rank, sequence_gt), dim=1)

                acc = 0
                for i in range (0, sequences_train.shape[0]):
                    acc += accuracy_score(rank[i].astype(int), rank_pred[i].cpu().detach().numpy().astype(int))

                acc = acc/sequences_train.shape[0]
                sequence_gt = torch.from_numpy(np.array(sequence_gt)).to(device)
                sorted_x = sorted_x[:,4:5,:,:]


                # loss = criterion(sorted_x, sequence_gt)
                og_distance = np.float32(np.array(og_distance))
                loss = criterion(torch.from_numpy(og_distance).to(device), scores)


                if args.wandb:
                    wandb.log({"loss": loss})

                if args.wandb:
                    wandb.log({"accuracy": acc})

                epochs.set_description(f"[{cnt+1}/{len(dataloader_train)}] training loss: {loss.item()}, accuracy: {acc}")

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

                    sequences_train = batch[:, :args.obs].reshape(-1, args.obs, batch.shape[2]//2, 2)
                    sequences_train_to_rank = sequences_train[:,4:5,:,:]
                    sequences_train_not_to_rank = sequences_train[:,:4,:,:] 

                    sorted_x, rank_pred, scores= model(sequences_train_not_to_rank,sequences_train_to_rank)

                    input = sequences_train_to_rank.cpu().detach().numpy()
                    sequence_gt, sorted_distance, og_distance, rank = get_output(input)

                    # sequences_train_not_to_rank = rank_to_coordinates(sequences_train_not_to_rank, rank)
                    # sequences_train_not_to_rank = torch.from_numpy(sequences_train_not_to_rank).to(device)
                    # sequence_gt = torch.from_numpy(np.array(sequence_gt)).to(device)
                    # sequences_train_gt = torch.cat((sequences_train_not_to_rank, sequence_gt), dim=1)

                    acc = 0
                    for i in range (0, sequences_train.shape[0]):
                        acc += accuracy_score(rank[i].astype(int), rank_pred[i].cpu().detach().numpy().astype(int))

                    acc = acc/sequences_train.shape[0]
                    sequence_gt = torch.from_numpy(np.array(sequence_gt)).to(device)
                    sorted_x = sorted_x[:,4:5,:,:]

                    # loss = criterion(sorted_x, sequence_gt)
                    og_distance = np.float32(np.array(og_distance))
                    loss = criterion(torch.from_numpy(og_distance).to(device), scores)

                        
                    if args.wandb:
                        wandb.log({"val loss": loss})    

                    if args.wandb:
                        wandb.log({"val accuracy": acc})        
                    
                    epochs.set_description(f"[{cnt+1}/{len(dataloader_train)}] val loss: {loss.item()}, val accuracy: {acc}")

                    running_loss += loss*batch.shape[0]
                    n += batch.shape[0]
                val_loss.append(running_loss/n)
                
            if args.use_scheduler:
                scheduler.step()

            if (val_loss[-1] < best_val[0]):
                best_val = [val_loss[-1], epoch]
                torch.save(model.state_dict(),args.model_path+"/order_nn/best_order_nn_"+str(args.reg)+"_"+str(args.lr)+".pth")
                print("new best val:", best_val[0])

    
        print("best val:", best_val)


    if args.mode =="test":

        if args.ordering1:
            euc = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)
            euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))


            pre_trained_model = SorterishWrapper(euc, args.reg, args.scale).to(device)



            criterion = nn.MSELoss()

            def test_order_nn(verbose=True):
                print('Loading validation dataset')
                val_loss = []
                val_acc = []

                dataset_val = nba_dataset.Dataset(dataset_path+"val.npy", fields, args.obs+args.preds)

                if verbose: print(f"Validation dataset size: {len(dataset_val)}")
                dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
                best_val = [1e10, None]
                # sw_gt = SorterishWrapper_GT()
                epochs = tqdm(range(args.n_epochs-1))


                for epoch in epochs:
                    running_loss = 0
                    running_acc = 0

                    n = 0        
                    with torch.no_grad():
                        running_loss = 0 
                        n = 0
                        for cnt,batch in enumerate(dataloader_val):
                            batch = batch.to(device)

                            # Data             
                            sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)
                            sequences_train_to_rank = sequences_train[:,4:5,:,:]
                            sequences_train_not_to_rank = sequences_train[:,:4,:,:]
                            
                            # Model 
                            sequences_pred, rank_pred, scores = pre_trained_model(sequences_train_not_to_rank, sequences_train_to_rank)
                            sequences_pred_evaluate = sequences_pred[:,4:5,:,:]
                            input = sequences_train_to_rank.cpu().detach().numpy()
                            sequence_gt, sorted_distance, og_distance, rank = get_output(input)


                            rank_pred_np = rank_pred.cpu().detach().numpy()   
                            rank_pred_np = rank_pred_np.astype(int)             
                            count_r = 0
                            for idx, num in enumerate(rank_pred_np):
                                if (rank[idx] == rank_pred_np[idx]).all() == True:
                                    count_r +=1
                                            
                            acc = count_r/len(rank)
                                    
                            # Loss 
                            loss = criterion(torch.from_numpy(sequence_gt).to(device), sequences_pred_evaluate) 

                            if args.wandb:
                                wandb.log({"test loss": loss})  

                            if args.wandb:
                                wandb.log({"test acc": acc})  
                                        
                            epochs.set_description(f"[{cnt+1}/{len(dataloader_val)}] val loss: {loss.item()}")

                            
                            running_acc += acc*batch.shape[0]
                            running_loss += loss*batch.shape[0]
                            n += batch.shape[0]

                        val_acc.append(running_acc/n)
                        val_loss.append(running_loss/n)

                    if (val_loss[-1] < best_val[0]):
                        best_val = [val_loss[-1], epoch]
                        print("new best val:", best_val[0], val_acc[-1])

            
                print("best val:", best_val)

        if args.ordering2:
            euc = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)
            # euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
            # euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"))
            euc.load_state_dict(torch.load(args.model_path+"/euc_best"))

            pre_trained_model = SorterishWrapper(euc, args.reg, args.scale).to(device)



            criterion = nn.MSELoss()

            def test_order_nn(verbose=True):
                print('Loading validation dataset')
                val_loss = []
                val_acc = []

                dataset_val = nba_dataset.Dataset(dataset_path+"val.npy", fields, args.obs+args.preds)

                dataset_reordered, final_index = ordering_data(dataset_val)

                if verbose: print(f"Validation dataset size: {len(dataset_val)}")
                dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
                dataloader_gt = DataLoader(dataset_reordered, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
                dataloader_index = DataLoader(final_index, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
                
                best_val = [1e10, None]
                # sw_gt = SorterishWrapper_GT()
                epochs = tqdm(range(args.n_epochs-1))


                for epoch in epochs:
                    running_loss = 0
                    running_acc = 0

                    n = 0        
                    with torch.no_grad():
                        running_loss = 0 
                        n = 0
                        for cnt,batch in enumerate(dataloader_val):

                            

                            # Load data GT
                            batch_gt = get_item_dataloader(dataloader_gt, cnt)
                            batch_gt = batch_gt.to(device)
                            # Load data index GT
                            batch_index = get_item_dataloader(dataloader_index, cnt)
                            batch_index = batch_index.to(device)

                            batch = batch.to(device)

                            # Data             
                            sequences_train = batch[:, :args.obs].view(-1, args.obs, batch.shape[2]//2, 2)
                            sequences_gt = batch_gt[:, args.obs:args.obs+args.preds].view(-1, args.preds, batch.shape[2]//2, 2)

                            # print('batch_index', batch_index.shape)
                            sequences_gt_index = batch_index[:, args.obs:args.obs+args.preds]

                            sequences_train_to_rank = sequences_train[:,4:5,:,:]
                            sequences_train_not_to_rank = sequences_train[:,:4,:,:]
                            
                            # Model 
                            sequences_pred, rank_pred, scores = pre_trained_model(sequences_train_not_to_rank, sequences_train_to_rank)

                            print(rank_pred.shape, sequences_gt_index.shape)
                            rank_pred_np = rank_pred.cpu().detach().numpy()   
                            rank_pred_np = rank_pred_np.astype(int)             
                            count_r = 0
                            acc = 0
                            for idx, num in enumerate(rank_pred_np):
                                if (sequences_gt_index[idx] == rank_pred_np[idx]).all() == True:
                                    count_r +=1
                                            
                            acc = count_r/len(rank)
                                    
                            # Loss 
                            loss = criterion(torch.from_numpy(sequences_gt).to(device), sequences_pred) 

                            if args.wandb:
                                wandb.log({"test loss": loss})  

                            if args.wandb:
                                wandb.log({"test acc": acc})  
                                        
                            epochs.set_description(f"[{cnt+1}/{len(dataloader_val)}] val loss: {loss.item()}")

                            
                            running_acc += acc*batch.shape[0]
                            running_loss += loss*batch.shape[0]
                            n += batch.shape[0]

                        val_acc.append(running_acc/n)
                        val_loss.append(running_loss/n)

                    if (val_loss[-1] < best_val[0]):
                        best_val = [val_loss[-1], epoch]
                        print("new best val:", best_val[0], val_acc[-1])

            
                print("best val:", best_val)


# main function
if __name__ == "__main__":


    if args.wandb:


        if args.euclidean_distance:
            wandb.login()
            wandb.init(
                project="NBA",
                entity="lscofano",
                tags=['euclidean', 'players = '+str(args.entities), "batch_size ="+str(args.batch_size), "lr ="+str(args.lr), "mode ="+args.mode],
                config=vars(args),
            )
        else:
                wandb.login()
                wandb.init(
                    project="NBA",
                    entity="lscofano",
                    tags=['order_nn', 'players = '+str(args.entities), "batch_size ="+str(args.batch_size), "lr ="+str(args.lr), "mode ="+args.mode, "reg ="+str(args.reg), "scale ="+str(args.scale)],
                    config=vars(args),
                )

    
    if args.euclidean_distance:
        print("Euclidean Distance")
        train_euclidean()  

    if args.order_nn:
        if args.mode == "train":
            train_order_nn()
            print("Training Order NN")
        elif args.mode == "test":
            test_order_nn()
            print("Testing Order NN")


    if args.wandb:
        wandb.finish()







