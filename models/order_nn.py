import torch
import torch.nn as nn
from fast_soft_sort.pytorch_ops import soft_rank, soft_sort


class SequenceScorer(nn.Module):
    def __init__(self, in_seq_len, players, channels, hidden=[128], **kwargs):
        super(SequenceScorer, self).__init__()
        self.players = players
        self.in_seq_len = in_seq_len
        self.input = in_seq_len * players * channels
        self.net = []
        self.last = in_seq_len * players * channels
        for h in hidden:
            self.net.append(
                nn.Sequential(
                    nn.Linear(self.last, h),
                    nn.PReLU()
                )
            )
            self.last = h
        self.net.append(nn.Linear(self.last, players))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        # print(self.players, self.in_seq_len)
        # print(x.shape)
        # print(self.input, self.last)#, in_seq_len , players , channels)
        x = self.net(x.reshape(-1, self.input))
        return x

class SorterishWrapper(nn.Module):
    def __init__(self,  scorer, regularization_strength=1, exp_round_scale=1):
        super(SorterishWrapper, self).__init__()
        self.scorer = scorer
        self.regularization_strength = regularization_strength
        self.exp_round_scale = exp_round_scale
        
    def intensity(self, x, scale=1):
    #     return 1 if x == 0 else 0
        return torch.exp(-(x/scale)**2)
    
    def forward(self, x_not_sort, x_to_sort):
        
        scores = self.scorer(x_to_sort)

        rank = soft_rank(scores.cpu(), regularization_strength=self.regularization_strength).to(x_to_sort.device)

        players = rank.shape[1]
        # base is a batch*N*N tensor with columns [1,2...player_cnt]
        base = torch.arange(1,players+1).to(x_to_sort.device).repeat(rank.shape[0],1).repeat_interleave(players).view(-1,players,players)

        # rank is repeated (along rows); computes rank-base (delta rank) and apply intensity fn
        perm_matrix = self.intensity(rank.repeat_interleave(players).view(-1,players,players).transpose(1,2)-base, scale=self.exp_round_scale)
        
        x = torch.cat((x_to_sort, x_not_sort), dim =1)

        sorted_x = torch.einsum("ijkl,imk->ijml", x, perm_matrix)
        # return sorted_x, scores, rank, base, perm_matrix
        return sorted_x, rank, scores


# # Modified Sorterish Wrapper with the flattening of the T time frames
# class SorterishWrapper_flat(nn.Module):
#     def __init__(self,  scorer, regularization_strength=1, exp_round_scale=1):
#         super(SorterishWrapper_flat, self).__init__()
#         self.scorer = scorer
#         self.regularization_strength = regularization_strength
#         self.exp_round_scale = exp_round_scale
        
#     def intensity(self, x, scale=1):
#     #     return 1 if x == 0 else 0
#         return torch.exp(-(x/scale)**2)
    
#     def forward(self, x_to_sort):

        
#         scores = self.scorer(x_to_sort)

#         rank = soft_rank(scores.cpu(), regularization_strength=self.regularization_strength).to(x_to_sort.device)

#         players = rank.shape[1]
#         # base is a batch*N*N tensor with columns [1,2...player_cnt]
#         base = torch.arange(1,players+1).to(x_to_sort.device).repeat(rank.shape[0],1).repeat_interleave(players).view(-1,players,players)

#         # rank is repeated (along rows); computes rank-base (delta rank) and apply intensity fn
#         perm_matrix = self.intensity(rank.repeat_interleave(players).view(-1,players,players).transpose(1,2)-base, scale=self.exp_round_scale)


#         sorted_x = torch.einsum("ijkl,imk->ijml", x_to_sort, perm_matrix)
#         # return sorted_x, scores, rank, base, perm_matrix
#         return sorted_x, rank, perm_matrix, scores


# class SorterishWrapper_supervised(nn.Module):
#     def __init__(self,  scorer, regularization_strength=1, exp_round_scale=1):
#         super(SorterishWrapper_supervised, self).__init__()
#         self.scorer = scorer
#         self.regularization_strength = regularization_strength
#         self.exp_round_scale = exp_round_scale
        
#     def intensity(self, x, scale=1):
#     #     return 1 if x == 0 else 0
#         return torch.exp(-(x/scale)**2)
    
#     def forward(self, x_not_sort, x_to_sort):

        
#         scores = self.scorer(x_to_sort)

#         rank = soft_rank(scores.cpu(), regularization_strength=self.regularization_strength).to(x_to_sort.device)

#         players = rank.shape[1]
#         # base is a batch*N*N tensor with columns [1,2...player_cnt]
#         base = torch.arange(1,players+1).to(x_to_sort.device).repeat(rank.shape[0],1).repeat_interleave(players).view(-1,players,players)

#         # rank is repeated (along rows); computes rank-base (delta rank) and apply intensity fn
#         perm_matrix = self.intensity(rank.repeat_interleave(players).view(-1,players,players).transpose(1,2)-base, scale=self.exp_round_scale)
        
#         x = torch.cat((x_to_sort, x_not_sort), dim =1)

#         sorted_x = torch.einsum("ijkl,imk->ijml", x, perm_matrix)
#         # return sorted_x, scores, rank, base, perm_matrix
#         return scores, sorted_x, rank


# class SorterishWrapper_GT(nn.Module):
#     def __init__(self, exp_round_scale=1):
#         super(SorterishWrapper_GT, self).__init__()

#         self.exp_round_scale = exp_round_scale
        
#     def intensity(self, x, scale=1):
#         return torch.exp(-(x/scale)**2)
    
#     def forward(self, x, rank):

#         players = rank.shape[1]
#         # base is a batch*N*N tensor with columns [1,2...player_cnt]
#         base = torch.arange(1,players+1).to(x.device).repeat(rank.shape[0],1).repeat_interleave(players).view(-1,players,players)
#         # rank is repeated (along rows); computes rank-base (delta rank) and apply intensity fn
#         perm_matrix = self.intensity(rank.repeat_interleave(players).view(-1,players,players).transpose(1,2)-base, scale=self.exp_round_scale)
#         sorted_x = torch.einsum("ijkl,imk->ijml", x, perm_matrix)

#         return sorted_x



# Class I used to test SoftRank and the approximation function in utils_func.py
class SorterishWrapper_test(nn.Module):
    def __init__(self, regularization_strength=1, exp_round_scale=1):
        super(SorterishWrapper_test, self).__init__()
        self.regularization_strength = regularization_strength
        self.exp_round_scale = exp_round_scale
        
    def intensity(self, x, scale=1):
    #     return 1 if x == 0 else 0
        return torch.exp(-(x/scale)**2)
    
    def forward(self, scores, x_to_sort):

        rank = soft_rank(scores.cpu(), regularization_strength=self.regularization_strength).to(x_to_sort.device)

        players = rank.shape[1]
        # base is a batch*N*N tensor with columns [1,2...player_cnt]
        base = torch.arange(1,players+1).to(x_to_sort.device).repeat(rank.shape[0],1).repeat_interleave(players).view(-1,players,players)

        # rank is repeated (along rows); computes rank-base (delta rank) and apply intensity fn
        perm_matrix = self.intensity(rank.repeat_interleave(players).view(-1,players,players).transpose(1,2)-base, scale=self.exp_round_scale)
        
        # x = torch.cat((x_to_sort, x_not_sort), dim =1)


        sorted_x = torch.einsum("ijkl,imk->ijml", x_to_sort, perm_matrix)
        # return sorted_x, scores, rank, base, perm_matrix
        return sorted_x, rank
