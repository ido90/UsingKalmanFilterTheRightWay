
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

def rand_range(a,b):
    return (b - a) * np.random.rand() + a

def rand_range_sym(d):
    return d * (2*np.random.rand()-1)

def simulate_data(n_targets=2000, x0=600, y0=600, v0=(30,90),
                  n_intervals=(4,7), int_len=(5,9), ar_sigma=2, at_sigma=15,
                  noise_r=1, noise_t=0.8*np.pi/180, save_base='Driving/data/', save='lidar'):
    X, Z = [], []
    for i in range(n_targets):
        x, y, vx, vy, zx, zy = [],[],[],[],[],[]
        # init state
        x.append(rand_range_sym(x0))
        y.append(rand_range_sym(y0))
        vtheta = rand_range(0,2*np.pi)
        v = rand_range(*v0)
        vx.append(v*np.cos(vtheta))
        vy.append(v*np.sin(vtheta))
        r = np.linalg.norm((x[-1],y[-1])) + np.random.normal(0, noise_r)
        theta = np.arctan2(y[-1], x[-1]) + np.random.normal(0, noise_t)
        zx.append(r*np.cos(theta))
        zy.append(r*np.sin(theta))
        # intervals
        n_ints = np.random.randint(*n_intervals)
        for interval in range(n_ints):
            interval_len = np.random.randint(*int_len)
            ar = np.random.normal(0, ar_sigma)
            at = rand_range_sym(at_sigma)
            for t in range(interval_len):
                # motion step
                v = np.linalg.norm((vx[-1],vy[-1]))
                ax = ar*vx[-1]/v - at*vy[-1]/v
                ay = ar*vy[-1]/v + at*vx[-1]/v
                x.append(x[-1]+vx[-1]+0.5*ax)
                y.append(y[-1]+vy[-1]+0.5*ay)
                vx.append(vx[-1]+ax)
                vy.append(vy[-1]+ay)
                # observation step
                r = np.linalg.norm((x[-1],y[-1])) + np.random.normal(0, noise_r)
                theta = np.arctan2(y[-1], x[-1]) + np.random.normal(0, noise_t)
                zx.append(r*np.cos(theta))
                zy.append(r*np.sin(theta))
        # save target
        XX = pd.DataFrame(dict(
            x = x,
            y = y,
            vx = vx,
            vy = vy,
        ))
        ZZ = pd.DataFrame(dict(
            zx = zx,
            zy = zy,
        ))
        X.append(XX)
        Z.append(ZZ)
    if save:
        with open(save_base+save+'.pkl', 'wb') as fd:
            pkl.dump((X,Z), fd)
    return X, Z

def load_data(save='lidar', save_base='Driving/data/'):
    with open(save_base+save+'.pkl', 'wb') as fd:
        X, Z = pkl.load(fd)
    return X, Z

def get_trainable_data(Z, X):
    # just convert dataframes to numpy arrays
    return [z.values.astype(np.double) for z in Z], [x.values.astype(np.double) for x in X]

def get_F():
    # x,y,vx,vy
    return torch.tensor([
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0],
        [0,0,0,1]
    ], dtype=torch.double)

def get_H():
    # x,y -> x,y,vx,vy
    return torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
    ], dtype=torch.double)

def z2x(z):
    return torch.cat((z, torch.zeros(2,dtype=torch.double)))

def loss_fun(v=False):
    return lambda pred, x: ((pred[:2+2*v]-x[:2+2*v])**2).sum()
