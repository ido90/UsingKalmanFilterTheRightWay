
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
from warnings import warn
import pickle as pkl
from time import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mp

import utils


MOT20_PATH = 'data/MOT20Labels/train/'

def load_ground_truth(base_path=MOT20_PATH, subsample=1):
    ''' https://arxiv.org/pdf/2003.09003.pdf '''
    columns = ('frame', 'target', 'x', 'y', 'w', 'h', 'flag1', 'obj_type', 'visibility')
    dd = pd.DataFrame(columns=('dir_id',)+columns)
    for dr in os.listdir(base_path):
        if os.path.isdir(base_path+dr):
            path = base_path+dr+'/gt/gt.txt'
            d = pd.read_csv(path, names=columns)
            d['dir_id'] = int(dr[-2:])
            dd = pd.concat((dd, d))
    dd.reset_index(drop=True, inplace=True)
    dd['x'] = dd['x'] + dd['w']/2
    dd['y'] = dd['y'] + dd['h']/2
    if subsample > 1:
        ids = (np.arange(len(dd)) % subsample) == 0
        dd = dd[ids]
    dd['track_start'] = (dd.target.diff()!=0).values + [False]
    dd['track_end'] = np.roll(dd['track_start'], -1)
    dd['unique_id'] = np.cumsum(dd.track_start)
    return dd

def get_trainable_data(dd):
    dd = [v.iloc[:, 3:7].values.astype(np.double) for k, v in dd.groupby('unique_id') if len(v)>2]
    Z = dd
    X = [
        np.concatenate((
            d,
            np.concatenate((d[1:2,:2]-d[0:1,:2], (d[2:,:2]-d[:-2,:2])/2, d[-1:,:2]-d[-2:-1,:2]), axis=0)
        ), axis=1).astype(np.double) for d in dd
    ]
    return Z, X

def plot_targets(dd, n=4):
    axs = utils.Axes(n, 4, axsize=(6,4))
    i0 = 0
    for a in range(n):
        i1 = i0 + np.where(dd.track_end[i0:])[0][0] + 1
        axs[a].plot(dd[i0:i1].x.values, dd[i0:i1].y.values, '.-')
        axs.labs(a, 'x', 'y')
        i0 = i1
    return axs

def get_F():
    # x,y,w,h,vx,vy
    return torch.tensor([
        [1,0,0,0,1,0],
        [0,1,0,0,0,1],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
    ], dtype=torch.double)

def get_H():
    # x,y,w,h -> x,y,w,h,vx,vy
    return torch.tensor([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
    ], dtype=torch.double)

def z2x(z):
    return torch.cat((z, torch.zeros(2,dtype=torch.double)))

def loss(pred, x):
    return ((pred[:2]-x[:2])**2).sum()
