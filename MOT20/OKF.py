
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from warnings import warn
import pickle as pkl
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mp
from torch import optim

import utils

class OKF(nn.Module):
    def __init__(self, dim_x, dim_z, framework='MOT20', title='KF', P0=1e3, Q0=2, R0=20,
                 F=None, H=None, H_fun=None, z2x=None, loss_fun=None, optimize=False):
        nn.Module.__init__(self)
        self.framework = framework
        self.title = title
        self.optimize = optimize
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.P0 = P0
        self.Q0 = Q0
        self.R0 = R0
        self.H = H # torch.zeros((self.dim_z, self.dim_x))
        self.H_fun = H_fun
        if self.H is None and self.H_fun is None:
            raise ValueError('Both H interfaces are disabled.')
        self.F = F
        if self.F is None:
            self.F = torch.eye(self.dim_x, dtype=torch.double)
        if self.optimize:
            self.Q_D = nn.Parameter((self.Q0 * (0.5 + torch.rand(self.dim_x, dtype=torch.double))).log(), requires_grad=True)
            self.R_D = nn.Parameter((self.R0 * (0.5 + torch.rand(self.dim_z, dtype=torch.double))).log(), requires_grad=True)
            self.Q_L = nn.Parameter(self.Q0/4 * torch.randn(self.dim_x * (self.dim_x-1) // 2, dtype=torch.double), requires_grad=True)
            self.R_L = nn.Parameter(self.R0/4 * torch.randn(self.dim_z * (self.dim_z-1) // 2, dtype=torch.double), requires_grad=True)
        else:
            self.Q_D = torch.zeros(self.dim_z, dtype=torch.double)
            self.R_D = torch.zeros(self.dim_z, dtype=torch.double)
            self.Q_L = torch.zeros(self.dim_x * (self.dim_x-1) // 2, dtype=torch.double)
            self.R_L = torch.zeros(self.dim_z * (self.dim_z-1) // 2, dtype=torch.double)
        self.z2x = z2x
        if self.z2x is None:
            self.z2x = lambda x:x
        self.loss_fun = loss_fun
        if self.loss_fun is None:
            self.loss_fun = lambda pred,x: ((pred-x)**2).sum()

        self.x = None
        self.z = None
        self.P = None
        self.fully_initialized = False

        self.init_state()

    def init_state(self):
        self.x = self.dim_x * [None]
        self.z = self.dim_z * [None]
        self.P = self.P0 * torch.eye(self.dim_x, dtype=torch.double)
        self.fully_initialized = False

    def reset_model(self):
        if self.optimize:
            self.Q_D = nn.Parameter((self.Q0 * (0.5 + torch.rand(self.dim_x, dtype=torch.double))).log(), requires_grad=True)
            self.R_D = nn.Parameter((self.R0 * (0.5 + torch.rand(self.dim_z, dtype=torch.double))).log(), requires_grad=True)
            self.Q_L = nn.Parameter(self.Q0/4 * torch.randn(self.dim_x * (self.dim_x-1) // 2, dtype=torch.double), requires_grad=True)
            self.R_L = nn.Parameter(self.R0/4 * torch.randn(self.dim_z * (self.dim_z-1) // 2, dtype=torch.double), requires_grad=True)
        else:
            self.Q_D = torch.zeros(self.dim_z, dtype=torch.double)
            self.R_D = torch.zeros(self.dim_z, dtype=torch.double)
            self.Q_L = torch.zeros(self.dim_x * (self.dim_x-1) // 2, dtype=torch.double)
            self.R_L = torch.zeros(self.dim_z * (self.dim_z-1) // 2, dtype=torch.double)

    def save_model(self, fname=None, base_path=None):
        if base_path is None: base_path = f'data/models/{self.framework}/'
        if fname is None: fname = self.title
        if fname.endswith('.m'): fname = fname[:-2]
        if self.optimize:
            # module state dict
            torch.save(self.state_dict(), base_path+fname+'.m')
        else:
            # noise estimation
            torch.save((self.Q_D, self.Q_L, self.R_D, self.R_L), base_path+fname+'.noise')

    def load_model(self, fname=None, base_path=None):
        if base_path is None: base_path = f'data/models/{self.framework}/'
        if fname is None: fname = self.title
        if fname.endswith('.m'): fname = fname[:-2]
        if self.optimize:
            # module state dict
            self.load_state_dict(torch.load(base_path + fname + '.m'))
        else:
            # noise estimation
            self.Q_D, self.Q_L, self.R_D, self.R_L = torch.load(base_path+fname+'.noise')

    def predict(self):
        if not self.fully_initialized:
            return
        self.x = mp(self.F, self.x)
        Q = OKF.get_SPD(self.Q_D, self.Q_L)
        self.P = mp(mp(self.F, self.P), self.F.T) + Q

    def update(self, z):
        # get observation
        self.z = torch.tensor(z)
        self.H = self.H if self.H_fun is None else self.H_fun(self.x, self.z)

        # get update operators
        R = OKF.get_SPD(self.R_D, self.R_L)
        Ht = self.H.T
        PHt = mp(self.P, Ht)
        self.S = mp(self.H, PHt) + R
        K = mp(PHt, self.S.inverse())

        # update P
        I_KH = torch.eye(self.P.shape[0]) - mp(K, self.H)
        # I_KH = torch.eye(self.P[i].shape[0]) - mp(K, self.H)
        # self.P[i] = mp(torch.eye(self.P[i].shape[0])-mp(K,self.H), self.P[i])
        self.P = mp(mp(I_KH, self.P), I_KH.T) + mp(mp(K, R), K.T)  # numeric stability: equivalent but more stable formula
        self.P = 0.5 * (self.P + self.P.T)  # numeric stability: force symmetric P

        # update x
        if self.fully_initialized:
            self.x = self.x + mp(K, self.z - mp(self.H, self.x))
        else:
            self.x = self.z2x(self.z)
            self.fully_initialized = True

    @staticmethod
    def get_SPD(D, L):
        n = len(D)
        A = D.exp().diag() # fill diagonal
        ids = torch.tril_indices(n, n, -1)
        A[ids[0, :], ids[1, :]] = L # fill below-diagonal
        return mp(A, A.T)

    @staticmethod
    def encode_SPD(A, eps=1e-6):
        n = A.shape[0]
        A = torch.cholesky(A+eps*torch.eye(n))
        D = A.diag()
        D = D.log()
        ids = torch.tril_indices(n,n,-1)
        L = A[ids[0,:],ids[1,:]]
        return D, L

    def estimate_noise(self, X, Z):
        # X, Z are assumed to be numpy arrays
        # Q
        X1 = torch.cat([torch.tensor(x[:-1]) for x in X], dim=0) # t
        X2 = torch.cat([torch.tensor(x[1:])  for x in X], dim=0) # t+1
        X1 = mp(self.F, X1.T).T  # prediction t+1|t
        Q = torch.tensor(np.cov((X1-X2).T.detach().numpy()))
        Q_D, Q_L = OKF.encode_SPD(Q)
        if self.optimize:
            with torch.no_grad():
                self.Q_D.copy_(Q_D)
                self.Q_L.copy_(Q_L)
        else:
            self.Q_D, self.Q_L = Q_D, Q_L

        # R
        Z = np.concatenate(Z, axis=0)
        H = [self.H_fun(torch.tensor(x), torch.tensor(z)) for x,z in zip(X,Z)] \
            if self.H_fun is not None else len(X)*[self.H]
        Xz = np.concatenate([mp(h,torch.tensor(x).T).T.detach().numpy() for x,h in zip(X,H)], axis=0)
        delta = Z - Xz
        R = torch.tensor(np.cov((delta).T))
        R_D, R_L = OKF.encode_SPD(R)
        if self.optimize:
            with torch.no_grad():
                self.R_D.copy_(R_D)
                self.R_L.copy_(R_L)
        else:
            self.R_D, self.R_L = R_D, R_L


#############################################################

def split_train_valid(X, Y, p=0.15, seed=9):
    n_valid = int(np.round(p*len(X)))
    np.random.seed(seed)
    ids_valid = set(list(np.random.choice(np.arange(len(X)), n_valid, replace=False)))
    Xt = [x for i,x in enumerate(X) if i not in ids_valid]
    Yt = [x for i,x in enumerate(Y) if i not in ids_valid]
    Xv = [x for i,x in enumerate(X) if i in ids_valid]
    Yv = [x for i,x in enumerate(Y) if i in ids_valid]
    return Xt, Yt, Xv, Yv

def print_train_summary(n_epochs, n_batches, early_stop, epoch, i_batch, valid_loss, T0, tit):
    print(f'[{tit:s}] Training done ({time() - T0:.0f} [s])')
    if early_stop:
        print(
            f'\tbest valid loss: {valid_loss:.0f};\tearly stopping:\t{epoch + 1:d}.{i_batch + 1:03d}/{n_epochs:d}.{n_batches:03d} ({100 * (epoch*n_batches + i_batch + 1) / (n_epochs * n_batches):.0f}%)')
    else:
        print(
            f'\tbest valid loss: {valid_loss:.0f};\tno early stopping:\t{n_epochs:d} epochs, {n_batches:d} batches, {n_epochs * n_batches:d} total iterations.')

def train(model, X, Y, split_data=None, p_valid=0.15, n_epochs=1, batch_size=10,
          lr=1e-2, lr_decay=0.5, lr_decay_freq=150, weight_decay=0.0,
          loss_after_pred=True, log_interval=300, reset_model=True, best_valid_loss=np.inf,
          verbose=2, valid_hor=8, to_save=True, save_best=True, **kwargs):
    # pre-processing
    Xt0, Yt0, Xv, Yv = split_train_valid(X, Y, p_valid) if split_data is None else split_data
    n_samples = len(Xt0)
    n_batches = n_samples // batch_size
    log_interval = log_interval // batch_size
    if lr_decay_freq is None: lr_decay_freq = n_batches

    if reset_model:
        model.reset_model()

    # estimate noise covariance
    model.estimate_noise(Y,X)

    if model.optimize:
        # initialize
        early_stop = False
        no_improvement_seq = 0
        e = 0
        # train monitor
        t = []
        losses = []
        RMSE = []
        # valid monitor
        t_valid = []
        losses_valid = []
        RMSE_valid = []

        model.train()
        optimizer = optim.Adam
        if weight_decay != 0:
            warn('Note: weight decay is known to cause instabilities in the tracker training.')
        params = model.parameters()
        o = optimizer(params, lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.StepLR(o, step_size=lr_decay_freq, gamma=lr_decay)

        # train
        if verbose >= 1:
            print(f'\nTraining {model.title:s}:')
            print(f'samples={len(Xt0):d}(t)+{len(Xv):d}(v)={len(X):d}; batch_size={batch_size:d}; ' + \
                  f'iterations={n_epochs}(e)x{n_batches}(b)={n_epochs * n_batches:d}.')
        T0 = time()
        b = 0
        for e in range(n_epochs):
            t0 = time()

            # shuffle batches
            ids = np.arange(len(Xt0))
            np.random.shuffle(ids)
            Xt = [Xt0[i] for i in ids]
            Yt = [Yt0[i] for i in ids]

            for b in range(n_batches):
                tt = e * n_batches + b
                x = [Xt[b*batch_size+i] for i in range(batch_size)]
                y = [Yt[b*batch_size+i] for i in range(batch_size)]

                loss_batch = train_step(x, y, model, o, loss_after_pred=loss_after_pred, **kwargs)

                t.append(tt+1)
                losses.append(loss_batch)
                RMSE.append(np.sqrt(loss_batch))

                if log_interval > 0 and ((tt % log_interval == 0) or tt==n_epochs*n_batches-1):
                    loss_batch = test_model(model, Xv, Yv, detailed=False, loss_after_pred=loss_after_pred)
                    model.train()

                    t_valid.append(tt+1)
                    losses_valid.append(loss_batch)
                    RMSE_valid.append(np.sqrt(loss_batch))

                    if verbose >= 2:
                        print(f'\t[{model.title:s}] {e + 1:02d}.{b + 1:04d}/{n_epochs:02d}.{n_batches:04d}:\t' + \
                              f'train_RMSE={RMSE[-1]:.2f}, valid_RMSE={RMSE_valid[-1]:.2f}   |   {time() - t0:.0f} [s]')

                    if len(losses_valid)>1 and np.min(losses_valid[:-1])<best_valid_loss:
                        best_valid_loss = np.min(losses_valid[:-1])
                    # best_valid_loss = np.min(losses_valid[:-1]) if len(losses_valid)>1 else np.inf
                    improved = loss_batch < best_valid_loss
                    if improved:
                        no_improvement_seq = 0
                        if to_save and save_best:
                            model.save_model(to_save if isinstance(to_save, str) else None)
                    else:
                        no_improvement_seq += 1
                    if no_improvement_seq >= valid_hor:
                        early_stop = True
                        break

                # update lr
                sched.step()

            if verbose >= 2:
                print(f'[{model.title:s}] Epoch {e + 1}/{n_epochs} ({time() - t0:.0f} [s])')

            if early_stop:
                break

        if verbose >= 1:
            print_train_summary(n_epochs, n_batches, early_stop, e, b, best_valid_loss, T0, model.title)

        res = pd.concat((
            pd.DataFrame(dict(model=len(t)*[model.title], t=t, group=len(t)*['train'],
                              loss=losses, RMSE=RMSE)),
            pd.DataFrame(dict(model=len(t_valid)*[model.title], t=t_valid, group=len(t_valid)*['valid'],
                              loss=losses_valid, RMSE=RMSE_valid))
        )).copy()

    else:
        res = pd.DataFrame({})

    if to_save:
        if save_best and model.optimize:
            model.load_model(to_save if isinstance(to_save, str) else None)
        else:
            model.save_model(to_save if isinstance(to_save, str) else None)

    res_valid = test_model(model, Xv, Yv, detailed=True, loss_after_pred=loss_after_pred).copy()

    return res, res_valid

def train_step(X, Y, model, optimizer, clip=1, loss_after_pred=False, optimize_per_target=False):
    # assign weights to errors (uniform over time-steps or uniform over targets)
    targets_lengths = np.array([len(x) for x in X])
    targets_weights = 1/targets_lengths if optimize_per_target else np.ones(len(X))
    targets_weights = targets_weights / np.sum(targets_weights*targets_lengths)

    optimizer.zero_grad()
    tot_loss = torch.tensor(0.)
    for x,y,w in zip(X,Y,targets_weights):
        model.init_state()
        for t in range(len(x)):
            xx = x[t,:]
            yy = y[t,:]

            model.predict()
            if loss_after_pred:
                loss = model.loss_fun(model.x, torch.tensor(yy)) if t>0 else torch.tensor(0.)

            model.update(xx)
            if not loss_after_pred:
                loss = model.loss_fun(model.x, torch.tensor(yy))

            tot_loss = tot_loss + w * loss

    tot_loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return tot_loss.item()

def test_model(model, X, Y, detailed=True, loss_after_pred=True, count_base=0, verbose=0):
    with torch.no_grad():
        model.eval()
        # per-step data
        targets = []
        times = []
        losses = []
        SE = []
        AE = []
        # per-batch data
        tot_loss = 0

        count = 0
        t0 = time()
        if verbose >= 1:
            print(f'\nTesting {model.title:s}:')
        for tar, (XX, YY) in enumerate(zip(X, Y)):
            model.init_state()
            for t in range(len(XX)):
                count += 1
                x = XX[t,:]
                y = YY[t,:]

                model.predict()
                if loss_after_pred:
                    loss = model.loss_fun(model.x, y) if t>0 else torch.tensor(0.)

                model.update(x)
                if not loss_after_pred:
                    loss = model.loss_fun(model.x, y)

                loss = loss.item()
                tot_loss += loss

                if detailed:
                    targets.append(count_base+tar)
                    times.append(t)
                    SE.append(loss)
                    AE.append(np.sqrt(loss))
                    losses.append(loss)

        if verbose >= 1:
            print(f'done.\t({time()-t0:.0f} [s])')
        if detailed:
            return pd.DataFrame(dict(
                model = len(times) * [model.title],
                target = targets,
                t = times,
                SE = SE,
                AE = AE,
                loss=losses,
            ))
        tot_loss /= count
        return tot_loss
