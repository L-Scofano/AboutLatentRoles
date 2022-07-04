import torch 
import torch.nn as nn
from fast_soft_sort.pytorch_ops import soft_rank, soft_sort

class SequenceScorer(nn.Module):
    def __init__(self, in_seq_len, players, channels, hidden=[128], **kwargs):
        super(SequenceScorer, self).__init__()
        self.players = players
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

class SorterishWrapper_GT(nn.Module):
    def __init__(self, exp_round_scale=1):
        super(SorterishWrapper_GT, self).__init__()

        self.exp_round_scale = exp_round_scale
        
    def intensity(self, x, scale=1):
        return torch.exp(-(x/scale)**2)
    
    def forward(self, x, rank):

        players = rank.shape[1]
        # base is a batch*N*N tensor with columns [1,2...player_cnt]
        base = torch.arange(1,players+1).to(x.device).repeat(rank.shape[0],1).repeat_interleave(players).view(-1,players,players)
        # rank is repeated (along rows); computes rank-base (delta rank) and apply intensity fn
        perm_matrix = self.intensity(rank.repeat_interleave(players).view(-1,players,players).transpose(1,2)-base, scale=self.exp_round_scale)
        sorted_x = torch.einsum("ijkl,imk->ijml", x, perm_matrix)

        return sorted_x


import torch
import torch.nn as nn
import math
import numpy as np




class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim, 
                 in_channels
    ):
        super(ConvTemporalGraphical,self).__init__()

        self.A = nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        
      
        


    def forward(self, x):

        
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        return x.contiguous()



class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels, # Input channels (x, y)
                 out_channels, # Output channels (x, y)
                 kernel_size,
                 stride,
                 time_dim, # Input time dimension (5 frames)
                 joints_dim, # Number of players 
                 dropout,
                 bias=True):
        
        super(ST_GCNN_layer,self).__init__()
        
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        
        self.gcn = ConvTemporalGraphical(time_dim,
                                        joints_dim, 
                                        in_channels) 
        
        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        
        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:

            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()

        

    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x) 
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        return x



class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        
        
        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)] 



            
        
        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):
        
        output= self.block(x)
        return output



class ModelVanilla(nn.Module):
    """ 
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 bias=True):
        
        super(ModelVanilla,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.st_gcnns=nn.ModuleList()
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns=nn.ModuleList()
        
      
        self.st_gcnns.append(ST_GCNN_layer(input_channels,64,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(64,32,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
            
        self.st_gcnns.append(ST_GCNN_layer(32,64,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
                                               
        self.st_gcnns.append(ST_GCNN_layer(64,input_channels,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))                                               
                
                
                # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))
        
            
        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())


        

    def forward(self, x):
        for gcn in (self.st_gcnns):
            x = gcn(x)
            
        x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)
        
        x=self.prelus[0](self.txcnns[0](x))
        
        for i in range(1,self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) +x # residual connection
            
        return x



class Model(nn.Module):
    """ 
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 reg,
                 scale,
                 pretrained = False,
                 bias=True):
        
        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.pretrained = pretrained
        # GCN modules
        self.st_gcnns=nn.ModuleList()
        # Order Net
        self.hh = [128, 64, 32]
        # TCN modules
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns=nn.ModuleList()



        self.st_gcnns.append(ST_GCNN_layer(input_channels,32,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(32,16,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
        
        if self.pretrained == False:

            # Order inputs
            self.ordernet = SorterishWrapper(SequenceScorer(1, joints_to_consider, input_channels, self.hh), reg, scale)

        
        

        self.st_gcnns.append(ST_GCNN_layer(16,64,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
                                               
        self.st_gcnns.append(ST_GCNN_layer(64,input_channels,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))                                               
                
                
        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))
        
            
        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())


        
    
    def forward(self, x):


        

        if self.pretrained == False:

            x = x.permute(0, 2, 3, 1) # (b, 5, 11, channels)
            x_to_sort = x[:,4:5,:,:] # (b, 1, 11, channels)
            x_not_sort = x[:,0:4,:,:] # (b, 4, 11, channels)

            x, rank, scores = self.ordernet(x_not_sort, x_to_sort)

        x = x.permute(0,3,1,2)


        for gcn in (self.st_gcnns):
            x = gcn(x)


        # x = x.permute(0, 2, 3, 1) # (b, 5, 11, channels)

        # x_to_sort = x[:,4:5,:,:] # (b, 1, 11, channels)
        # x_not_sort = x[:,0:4,:,:] # (b, 4, 11, channels)

        # x, rank = self.ordernet(x_not_sort, x_to_sort)

        # x = x.permute(0,3,1,2)

        # for gcn in (self.st_gcnns)[1:]:
        #     x = gcn(x)


        x = x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)
        
        x = self.prelus[0](self.txcnns[0](x))
        
        for i in range(1,self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x # residual connection

        if self.pretrained == False:
            return x, rank, scores
        else:
            return x






