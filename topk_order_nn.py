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
    


euc = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim).to(device)
euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
pre_trained_model = SorterishWrapper(euc, args.reg, args.scale).to(device)
criterion = nn.MSELoss()

def test_order_nn(verbose=True):
    print('Loading validation dataset')
    val_loss = []
    val_acc = []
    top_1_acc = []
    top_5_acc = []
    top_3_acc = []


    dataset_val = nba_dataset.Dataset(dataset_path+"val.npy", fields, args.obs+args.preds)

    if verbose: print(f"Validation dataset size: {len(dataset_val)}")
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    best_val = [1e10, None]
    # sw_gt = SorterishWrapper_GT()
    epochs = tqdm(range(0,1))


    for epoch in epochs:
        running_loss = 0
        running_acc = 0
        running_top1_acc = 0
        running_top5_acc = 0
        running_top3_acc = 0

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
                top1 = 0.0
                top5 = 0.0
                top3 = 0.0
                for idx, num in enumerate(rank_pred_np):
                    
                    gt = rank[idx]
                    predicted = rank_pred_np[idx]

                    if (gt[:-1] == predicted[:-1]).all() == True:
                        count_r +=1
                    if gt[0] == predicted[0]:
                        top1 +=1

                    if (gt[:5] == predicted[:5]).all() == True:
                        top5 +=1
                    #top 3
                    if (gt[:3] == predicted[:3]).all() == True:
                        top3 +=1

                top1_acc_temp = top1/len(rank)      
                top5_acc_temp = top5/len(rank)
                top3_acc_temp = top3/len(rank)
                acc = count_r/len(rank)


                running_top1_acc += top1_acc_temp*batch.shape[0]
                running_top5_acc += top5_acc_temp*batch.shape[0]    
                running_top3_acc += top3_acc_temp*batch.shape[0]
                running_acc += acc*batch.shape[0]
                n += batch.shape[0]

            val_acc.append(running_acc/n)
            top_1_acc.append(running_top1_acc/n)
            top_5_acc.append(running_top5_acc/n)
            top_3_acc.append(running_top3_acc/n)



        print("Top 10 accuracy:", np.mean(val_acc))
        print("Top 1 accuracy:", np.mean(top_1_acc))
        print("Top 5 accuracy:", np.mean(top_5_acc))
        print("Top 3 accuracy:", np.mean(top_3_acc))


# main function
if __name__ == "__main__":
    test_order_nn()



    # top1 = 0.0
    # top5 = 0.0    
    # class_probs = model.predict(x)
    # for i, l in enumerate(labels):
    #     class_prob = class_probs[i]
    #     top_values = (-class_prob).argsort()[:5]
    #     if top_values[0] == l:
    #         top1 += 1.0
    #     if np.isin(np.array([l]), top_values):
    #         top5 += 1.0

    # print("top1 acc", top1/len(labels))
    # print("top1 acc", top5/len(labels))