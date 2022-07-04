import numpy as np
import torch
from models.order_nn import *
from utils.parser import args



# def scores_to_rank(scores):
#     kv = {v:i for i,v in enumerate(sorted(range(len(scores)), key=lambda x: scores[x]))}
#     return [kv[i]+1 for i in range(len(scores))]

# def get_output(input):
#     output = []
#     sorted_distance = []
#     og_distance = []
#     rank = []
#     for s in input: 
#         ball = s[-1][-1]
#         # Original Distance
#         o = [*map(lambda x: np.linalg.norm(s[0][x]-ball), range(s.shape[1]))]
#         og_distance.append(o)

#         # Sorted Distance
#         d = sorted(o)
#         sorted_distance.append(d)

#         # Ranking based on Distance
#         o = scores_to_rank(o)
#         oo = s*0
#         rank.append(o)


#         for i,k in enumerate(o):
#             oo[:,k-1] = s[:,i]
#         output.append(oo.tolist())

#     rank = np.array(rank)
#     output = np.float32(np.array(output))
#     return output, sorted_distance, og_distance, rank


# def rank_to_coordinates(input, rank):

#     output = []

#     for index, s in enumerate(input): 

#         # Ranking based on Distance
#         o = rank[index]
#         oo = s*0

#         for i,k in enumerate(o):
#             oo[:,k-1] = s[:,i]
#         output.append(oo.tolist())

#     output = np.float32(np.array(output))
#     return output



def scores_to_rank(scores):
    kv = {v:i for i,v in enumerate(sorted(range(len(scores)), key=lambda x: scores[x]))}
    return [kv[i]+1 for i in range(len(scores))]

def get_output(input):
    output = []
    sorted_distance = []
    og_distance = []
    rank = []

    for s in input: 
        
        # Original Distance
        o = [*map(lambda x: np.linalg.norm(s[0][x]-s[0][-1]), range(s.shape[1]))]
        og_distance.append(o)

        # Sorted Distance
        d = sorted(o)
        sorted_distance.append(d)

        # Ranking based on Distance
        o = scores_to_rank(o)
        oo = s*0
        rank.append(o)


        for i,k in enumerate(o):
            oo[:,k-1] = s[:,i]

        # convert tensor to numpy
        oo = np.roll(oo, -1, axis=1)   
         
        output.append(oo.tolist())

    rank = np.array(rank)
    output = np.float32(np.array(output))
    
    return output, sorted_distance, og_distance, rank


def rank_to_coordinates(input, rank):

    output = []
    input = input.cpu().detach().numpy()


    for index, s in enumerate(input): 

        # Ranking based on Distance
        o = rank[index]
        oo = s*0

        for i,k in enumerate(o):
            oo[:,k-1] = s[:,i]

        
        oo = np.roll(oo, -1, axis=1)   
        
        output.append(oo.tolist())

    output = np.float32(np.array(output))
    return output


def ordering_data(dataset):
    for i in range(len(dataset)):
        action = dataset[i]
        # Number of players 
        players = (action.shape[1]//2)

        ball_dists = []
        for j in range(players-1):
            dist_j = (action[4,2*j] - action[4,-2])**2 + (action[4,2*j+1] - action[4,-1])**2
            ball_dists.append(dist_j)
        reordering_indices = np.argsort(np.array(ball_dists))
        new_action = np.zeros(action.shape)
        for j,k in enumerate(reordering_indices):
            new_action[:,2*j:2*j+2] = action[:,2*k:2*k+2]

        start = action.shape[1]-2
        end = action.shape[1]
        new_action[:,start:end] = action[:,start:end]
        dataset[i] = torch.tensor(new_action)
    return dataset

# main function 
if __name__ == "__main__":

    # set random seed for reproducibility pytorch
    torch.manual_seed(123)


    criterion = torch.nn.MSELoss()



    if args.pre_trained:
        # create a random tensor with shape (1, 5, 11, 2)
        # input = torch.randn(args.batch_size, 5, 11, 2)
        # x_not_sort = input[: , :4 , :, :]
        # input = input[: , 4:5 , :, :]
        input = torch.randn(args.batch_size, 1, 11, 2)
        input = input.reshape(-1, 1, 11, 2)
        # x_not_sort = input[: , :4 , :, :]
        # input = input[: , 4:5 , :, :]

    else:
        # create a random tensor with shape (1, 1, 11, 2)
        input = torch.randn(args.batch_size, 1, 11, 2)


    # ------> Test Euclidean Distance

    # call function get_output
    output, sorted_distance, og_distance, rank = get_output(input)
    og_distance = torch.from_numpy(np.float32(og_distance))

    print('input:', input)
    print()
    print('output:', output)
    print()
    print('sorted_distance:', sorted_distance)
    print()
    print('og_distance:', og_distance)
    print()
    print('rank:', rank)
    print()

    

    # ------> Test Order Neural Network

    if args.pre_trained:

        euc = SequenceScorer(1, args.entities, args.input_dim, args.hidden_dim)
        predicted_euclidean = euc(input)
        print('Euclidean Distance: ',criterion(predicted_euclidean, og_distance))
        print()

        # euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"+args.model+'_players_'+str(args.entities)+"_batch_size_"+str(args.batch_size)+"_lr_"+str(args.lr)))
        euc.load_state_dict(torch.load(args.model_path+"/euclidean/euc_best"))
        pre_trained_model = SorterishWrapper_flat(euc, args.reg, args.scale)

        permuted_coordinates, rank_pred, permuted_matrix = pre_trained_model(input)



    else:
        sw_gt = SorterishWrapper_test(regularization_strength= 0.001, exp_round_scale=0.01)
        permuted_coordinates, rank_pred = sw_gt(og_distance, input)

    print('Permuted matrix: ', permuted_matrix)
    print()
    print('permuted_coordinates:', permuted_coordinates)
    print()
    print('rank_pred:', rank_pred)
    print()


    # Accuracy
    rank_pred_np = rank_pred.cpu().detach().numpy()   
    rank_pred_np = rank_pred_np.astype(int)             
    count_r = 0
    for idx, num in enumerate(rank_pred_np):
        if (rank[idx] == rank_pred_np[idx]).all() == True:
            count_r +=1      
    acc = count_r/len(rank)


    # Test MSE and ACCURACY    
    output = torch.from_numpy(np.float32(output))
    loss = criterion(permuted_coordinates, output)
    print('loss:', loss)
    print()
    print('acc:', acc)
    print()
