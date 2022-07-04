from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import os
import json


CLOCK_GAME = [0]
CLOCK_SHOT = [1]
BALL_XY = [2,3]
BALL_XYZ = BALL_XY+[4]
TEAM_HOME = [5+i for i in range(5*2)]
TEAM_AWAY = [5+5*2+i for i in range(5*2)]

class Dataset(Dataset):
    def __init__(self, path_npy, fields, full_seq_len, skip_prefix_frames=0):
        # game_clock, shot_clock, ball(x,y,z), first_team pos (x,y), non_first_team pos (x,y)
        self.path_npy = path_npy
        self.full_seq_len = full_seq_len
        self.data = None
        
        dataset = np.load(path_npy)
        self.data = dataset[:, skip_prefix_frames:skip_prefix_frames+full_seq_len, fields]
        assert(self.data.shape[1] == full_seq_len)

        self.data = torch.tensor(self.data).float()

    def __len__(self):
        return self.data.shape[0]

    def __shape__(self):
        return self.data.shape

    def __getitem__(self, item):
        return self.data[item]
    
    def __setitem__(self, item, newvalue):
        self.data[item] = newvalue