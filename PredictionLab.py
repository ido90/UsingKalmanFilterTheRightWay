'''
Written by Ido Greenberg, 2020
'''

from pathlib import Path
import pickle as pkl
from time import time
import multiprocessing as mp
import os
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import seaborn as sns

import torch
import torch.optim as optim
from torch import autograd

import utils
import TrackingLab
import NeuralTrackers as NT

from sys import platform
if platform.startswith('linux'):
    import resource
    tmp = list(resource.getrlimit(resource.RLIMIT_NOFILE))
    # print('Max open files before:', tmp)
    tmp[0] = min(8*tmp[0], 32768)
    tmp[1] = max(tmp)
    resource.setrlimit(resource.RLIMIT_NOFILE, tmp)
    # tmp = resource.getrlimit(resource.RLIMIT_NOFILE)
    # print('Max open files after:', tmp)

DATA_PATH = Path('data/')

LINE_SCENARIOS = scenarios = dict(
    line_noisy_accless = dict(
        targets = dict(
            n_episodes = 4000,
            dt = 1,
            acc = 0,
            n_targets_args = 1,
            init_args=dict(t0=10, dt=2, X0=(0,0,0), dx=200, V0=(100,45,80), dV=(0,0,0)),
            n_turns_args = 0,
            line_args = dict(p_acc=1, t_mean=10, t_sigma=1.5),
            turn_args = dict(p_left=0.5, a_mean=45, a_sigma=10),
            seeds = None,
            title = 'Line',
        ),
        radar = dict(
            noise_factor = 0.5,
            FAR = 0,
        ),
    ),
    line_noisy_acc = dict(
        targets = dict(
            n_episodes = 4000,
            dt = 1,
            acc = 40,
            n_targets_args = 1,
            init_args=dict(t0=10, dt=2, X0=(0,0,0), dx=200, V0=(100,45,80), dV=(0,0,0)),
            n_turns_args = 0,
            line_args = dict(p_acc=1, t_mean=10, t_sigma=1.5),
            turn_args = dict(p_left=0.5, a_mean=45, a_sigma=10),
            seeds = None,
            title = 'Line',
        ),
        radar = dict(
            noise_factor = 0.5,
            FAR = 0,
        ),
    )
)

#########   PREPARE DATA   #########

def create_experiment(scenarios=None, title='pred_lines_data', load=True):
    if scenarios is None: scenarios = LINE_SCENARIOS
    E = TrackingLab.Experiment(title=title)
    if load:
        E.load_data(load if isinstance(load,str) else None)
    else:
        E.set_scenarios(scenarios)
        print(E.scenarios)
        E.generate_scenarios()
    return E

def split_data(E=None, df_episodes=None, df_targets=None, train_perc_per_scenario=None,
               default_train_perc=0.7, full_episodes=True):
    # input preprocessing
    if df_episodes is None: df_episodes = E.meta_episodes
    if df_targets is None: df_targets = E.meta_targets
    if train_perc_per_scenario is None:
        train_perc_per_scenario = {sc:default_train_perc for sc in np.unique(df_episodes.scenario)}

    # initialization
    df_episodes['group'] = 0
    df_targets['group'] = 0

    # do split
    if full_episodes:
        for sc, train_perc in train_perc_per_scenario.items():
            ids = (df_episodes.scenario == sc)
            n_tot = int(ids.sum())
            ids = np.where(ids)[0]
            n_train = int(np.round((1-train_perc) * n_tot))
            # test_episodes = np.random.choice(df_episodes.loc[ids, 'episode'], n_train, replace=False)
            test_ids = np.random.choice(ids, n_train, replace=False)
            test_episodes = set(df_episodes.loc[test_ids, 'episode'])
            df_episodes.loc[test_ids, 'group'] = 1
            df_targets.loc[(df_targets.scenario==sc) & (df_targets.episode.isin(test_episodes)), 'group'] = 1
    else:
        raise NotImplementedError()

def get_group_data(E, group=0, hue='target_class', shuffle=True, offset=0):
    dd = E.meta_targets
    data = [E.targets[sc][ep][tar].copy()
            for sc,ep,tar,grp in zip(dd.scenario,dd.episode,dd.target,dd.group) if grp==group]
    x_cols = [c for c in data[0].columns if c.startswith('z_')]
    if offset > 0:
        X = [d[x_cols].values[:-offset,:] for d in data]
    else:
        X = [d[x_cols].values[:,:] for d in data]
    y_cols = [c for c in data[0].columns if c.startswith('x_')]
    Y = [d[y_cols].values[offset:, :] for d in data]
    scenarios = dd[dd.group==group][hue].values
    if shuffle:
        ids = np.arange(len(X))
        np.random.shuffle(ids)
        X = [X[i] for i in ids]
        Y = [Y[i] for i in ids]
        scenarios = [scenarios[i] for i in ids]
    return X, Y, scenarios

def save_data(X, Y, scenarios, base=DATA_PATH/'XY', fname='tmp'):
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    fpath = base/fname
    with open(fpath, 'wb') as fd:
        pkl.dump((X,Y,scenarios), fd)

def load_data(base=DATA_PATH/'XY', fname='tmp'):
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    fpath = base/fname
    with open(fpath, 'rb') as fd:
        X,Y,scenarios = pkl.load(fd)
    return X, Y, scenarios

def extract_times(X, Y, dimX=4, dimY=6, eps=1e-9):
    times = []
    if X.shape[1] > dimX:
        times.append(X[:,0])
        X = X[:,1:]
    if Y.shape[1] > dimY:
        times.append(Y[:,0])
        Y = Y[:,1:]
    if len(times) > 0:
        if len(times) > 1:
            if np.sum(np.abs(times[0]-times[1])) > eps:
                warn('Inconsistent time records between observations and underlying states.')
                print(len(times[0]), np.sum(np.abs(times[0]-times[1])))
        dts = np.concatenate(([0], np.diff(times[0])))
    else:
        dts = len(X) * [None]
    return X, Y, dts


#########   TRAIN   #########

def split_train_valid(X, Y, p=0.15, seed=9):
    n_valid = int(np.round(p*len(X)))
    np.random.seed(seed)
    ids_valid = set(list(np.random.choice(np.arange(len(X)), n_valid, replace=False)))
    Xt = [x for i,x in enumerate(X) if i not in ids_valid]
    Yt = [x for i,x in enumerate(Y) if i not in ids_valid]
    Xv = [x for i,x in enumerate(X) if i in ids_valid]
    Yv = [x for i,x in enumerate(Y) if i in ids_valid]
    return Xt, Yt, Xv, Yv

def train_step(X, Y, model, optimizer, clip=1, mse_after_pred=False):
    # assign weights to errors (uniform over time-steps or uniform over targets)
    targets_lengths = np.array([len(x) for x in X])
    targets_weights = 1/targets_lengths if model.optimize_per_target else np.ones(len(X))
    targets_weights = targets_weights / np.sum(targets_weights*targets_lengths)

    optimizer.zero_grad()
    tot_loss = torch.tensor(0, dtype=model.precision).to(model.device)
    NLL = 0
    MSE = 0
    MAE = 0
    for x,y,w in zip(X,Y,targets_weights):
        x, y, dts = extract_times(x, y)
        model.init_state()
        for t in range(len(x)):
            xx = x[t,:]
            yy = y[t,:]
            dt = dts[t]
            model.predict(dt=dt)
            pred_loss, nll = model.loss_pred(yy)
            if mse_after_pred:
                if t > 0:
                    update_loss, mse = model.loss_update(yy)
                else:
                    update_loss, mse = torch.tensor(0.0), torch.tensor(0.0)
                model.do_update(xx)
            else:
                model.do_update(xx)
                update_loss, mse = model.loss_update(yy)
            tot_loss = tot_loss + w * (pred_loss + update_loss)
            NLL += w * nll.item()
            MSE += w * mse.item()
            MAE += w * np.sqrt(mse.item())

    tot_loss.backward()
    # TODO once in a while there's error on backward() for unknown reason.
    # I cannot reconstruct the error when rerunning from what seems like the same position (in terms of weights and data).
    # Since the error does not repeat, it is possible to simply catch the error using try/except and go on with the training in a recursive way.
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return tot_loss.item(), NLL, MSE, MAE

def print_train_summary(n_epochs, n_batches, early_stop, epoch, i_batch, valid_loss, T0, tit):
    print(f'[{tit:s}] Training done ({time() - T0:.0f} [s])')
    if early_stop:
        print(
            f'\tbest valid loss: {valid_loss:.0f};\tearly stopping:\t{epoch + 1:d}.{i_batch + 1:03d}/{n_epochs:d}.{n_batches:03d} ({100 * (epoch*n_batches + i_batch + 1) / (n_epochs * n_batches):.0f}%)')
    else:
        print(
            f'\tbest valid loss: {valid_loss:.0f};\tno early stopping:\t{n_epochs:d} epochs, {n_batches:d} batches, {n_epochs * n_batches:d} total iterations.')

def train_multi_phase(model, X, Y, p_valid=0.15, base_phase=True, Q_phase=False, tune_phase=True, verbose=1,
                      batch_size=10, batch_fac=2, phase_2_data_fac=1, phase_3_data_fac=1, out_losses=('RMSE','loss'),
                      lr=1e-2, lr_fac=0.3, lr_decay_freq=150, **kwargs):
    NN = bool(list(model.parameters()))
    data = split_train_valid(X, Y, p_valid)
    # phase I
    if verbose >= 1:
        print(f'[{model.title}] PHASE I:')
    if Q_phase:
        if model.dynamic_Q:
            model.freeze_Q_pred()
        if model.Rnet is not None:
            model.freeze_R_pred()
        if model.Hnet is not None:
            model.freeze_H_pred()
    if base_phase:
        train_res, valid_res = train(model, X, Y, split_data=data, batch_size=batch_size, lr=lr, lr_decay_freq=lr_decay_freq, **kwargs)
    else:
        train_res, valid_res = None, None
    # phase II
    if Q_phase and NN:
        if verbose >= 1:
            print(f'[{model.title}] PHASE II:')
        # train only Q-prediction
        for p in model.parameters():
            p.requires_grad = False
        if model.dynamic_Q:
            model.unfreeze_Q_pred()
            for p in model.Qnet.parameters():
                p.requires_grad = True
        if model.Rnet is not None:
            model.unfreeze_R_pred()
            for p in model.Rnet.parameters():
                p.requires_grad = True
        if model.Hnet is not None:
            model.unfreeze_H_pred()
            for p in model.Hnet.parameters():
                p.requires_grad = True
        # cut training shorter (through number of samples, since we're already at a merely single epoch)
        smaller_data = data
        if phase_2_data_fac < 1:
            n_train = len(data[0])
            n_smaller = int(phase_2_data_fac * n_train)
            smaller_data = (data[0][:n_smaller],data[1][:n_smaller],data[2],data[3])
        # train
        res = train(model, X, Y, split_data=smaller_data, reset_model=False, Q_phase=True,
                    batch_size=batch_fac*batch_size, lr=lr, lr_decay_freq=int(0.7*lr_decay_freq), **kwargs)
        # merge res
        if train_res is not None:
            res[0]['t'] = res[0]['t'] + train_res['t'].values[-1]
        train_res = res[0] if train_res is None else pd.concat((train_res, res[0]))
        valid_res = res[1]
    # phase III
    if tune_phase and NN:
        if verbose >= 1:
            print(f'[{model.title}] PHASE III:')
        # train all params
        if Q_phase:
            for p in model.parameters():
                p.requires_grad = True
        # cut training shorter (through number of samples, since we're already at a merely single epoch)
        smaller_data = data
        if phase_3_data_fac < 1:
            n_train = len(data[0])
            n_smaller = int(phase_3_data_fac * n_train)
            smaller_data = (data[0][:n_smaller],data[1][:n_smaller],data[2],data[3])
        # test for improvement wrt previous phase
        best_valid_loss = np.inf if train_res is None else train_res[train_res.group=='valid'].loss.min()
        # train
        res = train(model, X, Y, split_data=smaller_data, reset_model=False, batch_size=batch_fac*batch_size,
                    best_valid_loss=best_valid_loss, lr=lr_fac*lr, lr_decay_freq=int(0.7*lr_decay_freq), **kwargs)
        # merge res
        if train_res is not None:
            res[0]['t'] = res[0]['t'] + train_res['t'].values[-1]
        train_res = res[0] if train_res is None else pd.concat((train_res, res[0]))
        valid_res = res[1]
    if train_res is None or not NN:
        best_valid_loss = len(out_losses) * [np.inf]
    else:
        best_valid_loss = [train_res[train_res.group=='valid'][out_loss].min() for out_loss in out_losses]
    return train_res, valid_res, best_valid_loss

def train(model, X, Y, scenarios=None, split_data=None, p_valid=0.15, n_epochs=1, batch_size=10,
          lr=1e-2, lr_decay=0.5, lr_decay_freq=150, weight_decay=0.0, Q_phase=False,
          mse_after_pred=False, log_interval=300, reset_model=True, best_valid_loss=np.inf,
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
    if model.use_R_estimation:
        Xe, Ye = (X,Y) if model.estimate_from_valid else (Xt0,Yt0)
        model.estimate_R(Xe, Ye, inplace=True)
    if model.use_Q_estimation:
        Xe, Ye = (X,Y) if model.estimate_from_valid else (Xt0,Yt0)
        model.estimate_Q(Ye, inplace=True)

    NN = bool(list(model.parameters()))

    if NN:

        if model.normalize_features:
            model.set_features_normalizer(Xt0, Yt0)

        # initialize
        early_stop = False
        no_improvement_seq = 0
        e = 0
        # train monitor
        t = []
        losses = []
        NLL = []
        RMSE = []
        MAE = []
        # valid monitor
        t_valid = []
        losses_valid = []
        NLL_valid = []
        RMSE_valid = []
        MAE_valid = []

        model.train()
        optimizer = optim.Adam
        if weight_decay != 0:
            warn('Note: weight decay is known to cause instabilities in the tracker training.')
        if Q_phase:
            params = []
            if model.Qnet is not None: params.extend(list(model.Qnet.parameters()))
            if model.Rnet is not None: params.extend(list(model.Rnet.parameters()))
            if model.Hnet is not None: params.extend(list(model.Hnet.parameters()))
        else:
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

                try:
                    loss_batch, NLL_batch, MSE_batch, MAE_batch = train_step(x, y, model, o, mse_after_pred=mse_after_pred, **kwargs)
                except:
                    print('An error has occurred during training step.')
                    print('Train step failed.')
                    print(model.title, to_save)
                    print('Q_phase =', Q_phase, ' lr =', lr)
                    print(f'Batch {e}.{b}: ', [ids[b*batch_size+i] for i in range(batch_size)])
                    model.save_model(to_save+'debug' if isinstance(to_save, str) else 'debug')
                    raise

                t.append(tt+1)
                losses.append(loss_batch)
                NLL.append(NLL_batch)
                RMSE.append(np.sqrt(MSE_batch))
                MAE.append(MAE_batch)

                if log_interval > 0 and ((tt % log_interval == 0) or tt==n_epochs*n_batches-1):
                    loss_batch, NLL_batch, MSE_batch, MAE_batch = \
                        test_model(model, Xv, Yv, detailed=False, mse_after_pred=mse_after_pred)
                    model.train()

                    t_valid.append(tt+1)
                    losses_valid.append(loss_batch)
                    NLL_valid.append(NLL_batch)
                    RMSE_valid.append(np.sqrt(MSE_batch))
                    MAE_valid.append(MAE_batch)

                    if verbose >= 2:
                        print(f'\t[{model.title:s}] {e + 1:02d}.{b + 1:04d}/{n_epochs:02d}.{n_batches:04d}:\t' + \
                              f'train_NLL={NLL[-1]:.2f}, valid_NLL={NLL_valid[-1]:.2f}   |   ' + \
                              f'train_MAE={MAE[-1]:.2f}, valid_MAE={MAE_valid[-1]:.2f}   |   {time() - t0:.0f} [s]')

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
                              loss=losses, NLL=NLL, RMSE=RMSE, MAE=MAE)),
            pd.DataFrame(dict(model=len(t_valid)*[model.title], t=t_valid, group=len(t_valid)*['valid'],
                              loss=losses_valid, NLL=NLL_valid, RMSE=RMSE_valid, MAE=MAE_valid))
        )).copy()

    else:
        res = pd.DataFrame({})

    if to_save:
        if save_best and NN:
            model.load_model(to_save if isinstance(to_save, str) else None)
        else:
            model.save_model(to_save if isinstance(to_save, str) else None)

    res_valid = test_model(model, Xv, Yv, scenarios_list=scenarios, detailed=True, mse_after_pred=mse_after_pred).copy()

    return res, res_valid

def train_model(q, m, X, Y, kwargs):
    try:
        out = train_multi_phase(m, X, Y, **kwargs)
    except:
        q.put((m.title, (pd.DataFrame(),pd.DataFrame(),np.inf)))
        print(f'Error in {m.title}, returning empty output.')
        raise
    q.put((m.title, out))

def train_models(models_args, X, Y, model_fname_prefix=None,
                 seeds=1, verbose=2, save_res=True, **kwargs):
    if isinstance(seeds, int): seeds = list(range(seeds))
    if model_fname_prefix is None and isinstance(save_res, str): model_fname_prefix = save_res
    models_args = [(NT.NeuralKF, a) if isinstance(a,dict) else a for a in models_args]

    t0 = time()

    if len(models_args) == 1 and len(seeds) == 1:
        fun, margs = models_args[0]
        margs['seed'] = seeds[0]
        m = fun(**margs)
        if 'Q_phase' not in kwargs:
            kwargs['Q_phase'] = ('dynamic_Q' in margs and margs['dynamic_Q'])
        out = train_models_undistributed(
            [m], X, Y, **kwargs,
            to_save=(f'{model_fname_prefix:s}_{m.title}' if model_fname_prefix else True)
        )
        return [m], out[0], out[1], None, None, None

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    models = []
    procs = []
    for i, (fun,margs) in enumerate(models_args):
        margs = margs.copy()
        title = margs['title']
        for j, seed in enumerate(seeds):
            # construct model object
            margs['seed'] = seed
            margs['title'] = f'{title}_s{seed:02d}'
            m = fun(**margs)
            models.append(m)
            # get training args
            args = dict(verbose=verbose-1)
            for k, v in kwargs.items():
                if isinstance(v, (tuple, list)):
                    args[k] = v[i]
                else:
                    args[k] = v
            if 'Q_phase' not in args:
                args['Q_phase'] = ('dynamic_Q' in margs and margs['dynamic_Q']) #or ('pred_H' in margs and margs['pred_H'])
            if 'to_save' not in args:
                args['to_save'] = f'{model_fname_prefix:s}_{m.title}' if model_fname_prefix else True

            # run process
            p = ctx.Process(target=train_model, args=(q, m, X, Y, args))
            procs.append(p)
            if verbose >= 3:
                print(f'Training {m.title:s}...')
            p.start()

    # wait for processes
    if verbose >= 1:
        print('Waiting for trainings...')
    dd = [q.get() for _ in range(len(procs))]
    names = [d[0] for d in dd]
    dd = [d[1] for d in dd]
    if verbose >= 1:
        print('Waiting for processes...')
    for i,p in enumerate(procs):
        if verbose >= 3:
            print(f'Waiting for {models[i].title:s}...')
        p.join()
    if verbose >= 1:
        print(f'Done.\t({time()-t0:.0f} [s])')

    # choose best over seeds
    if verbose >= 2:
        print('Best validation losses:')
    d_t, d_v, d_t_all, d_v_all, chosen_models, all_losses = [], [], [], [], [], {}
    for i, (fun,margs) in enumerate(models_args):
        title = margs['title']
        if verbose >= 2: print(f'\t{title}:', end='')
        losses = []
        ids = [j for j in range(len(names)) if names[j][:-4]==title]
        best_seed = ids[0]
        best_loss = np.inf
        for j in ids:
            nm = names[j]
            seed = int(nm[-2:])
            d_t_all.append(dd[j][0])
            d_v_all.append(dd[j][1])
            loss = dd[j][2]
            if isinstance(loss, (tuple,list)): loss = loss[0]
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_seed = j
            if verbose >= 2: print(f' {loss:.0f} ({nm}),', end='')
        if verbose >= 2: print('\b')

        all_losses[title] = losses
        d_t.append(dd[best_seed][0])
        d_v.append(dd[best_seed][1])
        for m in models:
            if m.title == names[best_seed]:
                break
        else:
            raise LookupError(names[best_seed])
        m.load_model(f'{model_fname_prefix:s}_{m.title}' if model_fname_prefix else None)
        m.title = title
        m.save_model(f'{model_fname_prefix:s}_{m.title}' if model_fname_prefix else None)
        chosen_models.append(m)

    # summarize results
    d_t = pd.concat(d_t).reset_index(drop=True)
    d_v = pd.concat(d_v).reset_index(drop=True)
    d_t_all = pd.concat(d_t_all).reset_index(drop=True)
    d_v_all = pd.concat(d_v_all).reset_index(drop=True)
    if save_res:
        fname = save_res if isinstance(save_res, str) else 'tmp'
        fname = DATA_PATH/f'train/{fname:s}.pkl'
        with open(fname,'wb') as fd:
            pkl.dump((d_t,d_v, d_t_all,d_v_all, all_losses), fd)

    return chosen_models, d_t, d_v, d_t_all, d_v_all, all_losses

def train_models_undistributed(models, X, Y, **kwargs):
    models_to_keep = [(bool(list(m.parameters())) or m.use_R_estimation or m.use_Q_estimation) for m in models]
    models = [m for i,m in enumerate(models) if models_to_keep[i]]
    dd = []
    for i,m in enumerate(models):
        args = {}
        for k, v in kwargs.items():
            if isinstance(v, (tuple, list)):
                args[k] = v[i]
            else:
                args[k] = v
        dd.append(train_multi_phase(m, X, Y, **args))
    dd1 = pd.concat([d[0] for d in dd])
    dd2 = pd.concat([d[1] for d in dd])

    dd1.reset_index(drop=True, inplace=True)
    dd2.reset_index(drop=True, inplace=True)
    return dd1, dd2

def train_summary(res, axs=None, ylim_quant=None, detailed=False, axsize=(7,4)):
    if len(res) == 0:
        warn('Empty train monitor.')
        return
    models = np.unique(res.model)
    n_models = len(models)
    if axs is None: axs = utils.Axes(3*2+detailed*3*n_models, 3, axsize=axsize, fontsize=14)
    a = 0

    for y in ('RMSE', 'NLL', 'loss'):
        sns.lineplot(data=res[res.group=='train'], x='t', hue='model', y=y, ax=axs[a+0])
        sns.lineplot(data=res[res.group=='valid'], x='t', hue='model', y=y, ax=axs[a+1])
        utils.labels(axs[a+0], 't', y, 'Train set', 16)
        utils.labels(axs[a+1], 't', y, 'Validation set', 16)
        axs[a+0].set_yscale('log')
        axs[a+1].set_yscale('log')
        if ylim_quant:
            axs[a+0].set_ylim((None,np.percentile(res[res.group=='train'][y], ylim_quant)))
            axs[a+1].set_ylim((None,np.percentile(res[res.group=='valid'][y], ylim_quant)))
        a += 2

    if detailed:
        for m in models:
            sns.lineplot(data=res[res.model==m], x='t', hue='group', y='NLL', ax=axs[a+0])
            sns.lineplot(data=res[res.model==m], x='t', hue='group', y='MAE', ax=axs[a+1])
            res['MSE/100'] = res.RMSE**2 / 100
            tmp = utils.pd_merge_cols(res[(res.model==m) & (res.group=='train')],
                                      ('MSE/100','NLL','loss'), ['loss'], 'metric')
            sns.lineplot(data=tmp, x='t', hue='metric', y='loss', ax=axs[a+2])
            for ax in (axs[a+0],axs[a+1],axs[a+2]):
                ax.set_title(m, fontsize=16)
                ax.set_xlabel(ax.get_xlabel(), fontsize=16)
                ax.set_ylabel(ax.get_ylabel(), fontsize=16)
            for ax in (axs[a + 0],axs[a + 1]):
                ax.set_yscale('log')
            a += 3

    plt.tight_layout()

    return axs


#########   TEST   #########

def distribute_input(X, Y, scenarios=None, n=None, shuffle=False):
    if n is None: n = os.cpu_count()
    if shuffle:
        raise NotImplementedError()
    N = len(X)
    XX = [X[i*N//n : (i+1)*N//n] for i in range(n)]
    YY = [Y[i*N//n : (i+1)*N//n] for i in range(n)]
    if scenarios is not None:
        scenarios = [scenarios[i*N//n : (i+1)*N//n] for i in range(n)]
    return XX, YY, scenarios

def test_models(models, X, Y, scenarios=None, distributed=True, n_cpus=None, models_per_batch=8, verbose=1, **kwargs):
    if scenarios is None or len(scenarios)!=len(X):
        scenarios = len(X) * ['unknown']
    if distributed:
        XX, YY, scenarios = distribute_input(X, Y, scenarios, n=n_cpus)
        n_threads = len(XX)
        count_base = np.concatenate((np.array([0]),np.cumsum([len(x) for x in XX])[:-1]))
        n_model_batches = int(np.ceil(len(models)/models_per_batch))
        print(f'{len(models):d} models are split to {n_model_batches:d} batches.')
        t0 = time()
        dd = pd.DataFrame()
        for j in range(n_model_batches):
            args = [[i, n_threads, models[j*models_per_batch:(j+1)*models_per_batch], XX[i], YY[i], scenarios[i],
                     count_base[i], verbose, kwargs]
                    for i in range(n_threads)]
            with mp.Pool(processes=n_cpus) as P:
                assert n_threads==P._processes, f'{n_threads} data bathces != {P._processes} threads.'
                if verbose >= 1:
                    print(f'Running {n_threads:d} threads...')
                d = P.map(test_models_thread, args)
                d = list(d)
                dd = pd.concat([dd]+d)
            if verbose >= 1:
                print(f'Finished models-batch {j+1:d}/{n_model_batches:d}.\t({time()-t0:.0f} [s])')
    else:
        dd = pd.concat([test_model(m, X, Y, scenarios_list=scenarios, detailed=True, verbose=verbose, **kwargs)
                        for m in models])
    dd.reset_index(drop=True, inplace=True)
    return dd

def test_models_thread(args):
    i, n, models, X, Y, scenarios, count_base, verbose, kwargs = args
    t0 = time()
    dd = pd.concat([test_model(m, X, Y, scenarios_list=scenarios, detailed=True, count_base=count_base, verbose=0, **kwargs)
                    for m in models])
    dd.reset_index(drop=True, inplace=True)
    if verbose >= 2:
        print(f'Thread {i:02d}/{n:02d} done.\t({time()-t0:.0f} [s])')
    return dd

def test_model(model, X, Y, scenarios_list=None, detailed=True, mse_after_pred=False, count_base=0, verbose=0):
    with torch.no_grad():
        model.eval()
        # per-step data
        scenarios = []
        targets = []
        times = []
        predicted_gaussian = []
        matched_gaussian = []
        NLL = []
        SE = []
        AE = []
        losses = []
        # per-batch data
        tot_loss = 0
        NLL_tot = 0
        MSE_tot = 0
        MAE_tot = 0

        count = 0
        t0 = time()
        if verbose >= 1:
            print(f'\nTesting {model.title:s}:')
        for tar, (XX, YY) in enumerate(zip(X, Y)):
            XX, YY, dts = extract_times(XX, YY)
            model.init_state()
            for t in range(len(XX)):
                count += 1
                x = XX[t,:]
                y = YY[t,:]
                dt = dts[t]

                model.predict(dt=dt)
                pred_loss, nll, g1 = model.loss_pred(y, detailed=True, force=True)
                if mse_after_pred:
                    if t > 0:
                        update_loss, mse, g2 = model.loss_update(y, detailed=True, force=True)
                    else:
                        update_loss, mse, g2 = torch.tensor(0.0), torch.tensor(0.0), 0
                    model.do_update(x)
                else:
                    model.do_update(x)
                    update_loss, mse, g2 = model.loss_update(y, detailed=True, force=True)

                pred_loss = pred_loss.item()
                update_loss = update_loss.item()
                nll = nll.item()
                mse = mse.item()

                loss = pred_loss + update_loss
                tot_loss += loss
                NLL_tot += nll
                MSE_tot += mse
                MAE_tot += np.sqrt(mse)

                if detailed:
                    scenarios.append('unknown' if scenarios_list is None else scenarios_list[tar])
                    targets.append(count_base+tar)
                    times.append(t)
                    predicted_gaussian.append(g1)
                    matched_gaussian.append(g2)
                    NLL.append(nll)
                    SE.append(mse)
                    AE.append(np.sqrt(mse))
                    losses.append(loss)

        if verbose >= 1:
            print(f'done.\t({time()-t0:.0f} [s])')
        if detailed:
            return pd.DataFrame(dict(
                model = len(times) * [model.title],
                scenario = scenarios,
                target = targets,
                t = times,
                predicted_gaussian = predicted_gaussian,
                matched_gaussian = matched_gaussian,
                NLL = NLL,
                SE = SE,
                AE = AE,
                loss=losses,
            ))
        tot_loss /= count
        NLL_tot /= count
        MSE_tot /= count
        MAE_tot /= count
        return tot_loss, NLL_tot, MSE_tot, MAE_tot

def multi_scenario_test(scenarios, models_args, seeds=1, rf='KF', do_train=True,
                        epochs=1, min_iters=None, n_train=None, verbose=2):
    res = pd.DataFrame()
    for title in scenarios:
        # load scenario
        Xtr, Ytr, tars_tr = load_data(fname=f'{title}_train')
        Xts, Yts, tars_tst = load_data(fname=f'{title}_test')
        if n_train is not None:
            if n_train > len(Xtr):
                warn('n_train is larger than original dataset:', n_train, len(Xtr))
            Xtr = Xtr[:n_train]
            Ytr = Ytr[:n_train]
            tars_tr = tars_tr[:n_train]
            title = f'{title}_n{n_train:04d}'

        print(f'\n{title}')

        if min_iters is not None:
            epochs = max(epochs, (10*min_iters)//len(Xtr))

        # train
        if do_train:
            models, train_res, valid_res, _, _, losses = \
                train_models(models_args, Xtr, Ytr, seeds=seeds, save_res=title, model_fname_prefix=title, n_epochs=epochs, batch_size=10,
                                  tune_phase=False, lr=1e-2, lr_decay=0.5, lr_decay_freq=150, verbose=verbose)
            if verbose>=1 and len(train_res)>0:
                train_summary(train_res, ylim_quant=90)

        else:
            models = [NT.NeuralKF(**a) for a in models_args]
            for m in models:
                m.load_model(f'{title}_{m.title}')

        # test
        test_res = test_models(models, Xts, Yts, tars_tst)
        if verbose >= 0:
            test_analysis(test_res, scores=('NLL','SE'), ref_model=rf, per_target=True, axargs=dict(W=2,axsize=(9,4)))
        test_res['tar_class'] = test_res['scenario']
        test_res['scenario'] = title
        res = pd.concat((res, test_res))

    res.to_pickle('data/multi_scenario_res.pkl')
    return res

def test_across_scenarios(scenarios, models_args, verbose=1):
    n_sc = len(scenarios)
    n_models = len(models_args)

    MSE = np.nan * np.ones((n_models, n_sc, n_sc))
    NLL = np.nan * np.ones((n_models, n_sc, n_sc))
    res = dict(model=[], train_scenario=[], test_scenario=[], MSE=[], NLL=[])

    for i, sci in enumerate(scenarios):
        if verbose >= 1:
            print(f'Running models trained on {sci}...')
        models = [NT.NeuralKF(**a) for a in models_args]
        for m in models:
            m.load_model(f'{sci}_{m.title}')

        for j, scj in enumerate(scenarios):
            X, Y, tars = load_data(fname=f'{scj}_test')
            tmp_res = test_models(models, X, Y, tars)

            for m, model in enumerate(models):
                nll = tmp_res[tmp_res.model==model.title].NLL.mean()
                mse = tmp_res[tmp_res.model==model.title].SE.mean()

                MSE[m][i][j] = mse
                NLL[m][i][j] = nll
                res['model'].append(model.title)
                res['train_scenario'].append(sci)
                res['test_scenario'].append(scj)
                res['MSE'].append(mse)
                res['NLL'].append(nll)

    res = pd.DataFrame(res)
    res = res.groupby(['train_scenario','test_scenario']).apply(lambda d: d.assign(MSE_best=d.MSE.min(), NLL_best=d.NLL.min())).reset_index(drop=True)
    res['MSE_rel'] = res.MSE / res.MSE_best
    res['NLL_rel'] = res.NLL / res.NLL_best

    return res, NLL, MSE

#########   ANALYSIS   #########

def res_per_target(res, groupby='model', scores=('NLL','SE','AE'), separate_scenarios=False):
    if separate_scenarios:
        return pd.concat([res_per_target(res[res.scenario==sc], groupby=groupby, scores=scores, separate_scenarios=False) \
                          for sc in res.scenario.unique()])

    res = res.groupby('target').apply(lambda dd:dd.groupby(groupby).apply(lambda d:pd.DataFrame(dict(
        scenario=[d.scenario.values[0]], target=[d.target.values[0]], model=[d.model.values[0]], wt=[len(d)],
        NLL=[d.NLL.mean()], SE=[d.SE.mean()], AE=[d.AE.mean()], loss=[d.loss.mean()],
        tar_class=d.tar_class.values[0] if 'tar_class' in d.columns else np.nan
    ))))
    for score in scores:
        res[score] = res[score] * res.wt / res.wt.mean()
    return res

def multi_scenario_analysis(res, res_tar=None, scenarios=None, models=None, axs=None, axsize=(6,3.8),
                            skip_t=1, hetro_models=None, print_table=False, savefig=None):
    if scenarios is None: scenarios = res.scenario.unique()
    if models is None: models = res.model.unique()
    if hetro_models is None: hetro_models = np.any([m.startswith('O') for m in models]) and np.any([not m.startswith('O') for m in models])
    if skip_t:
        res = res[res.t>=skip_t]
    if res_tar is None:
        res_tar = res_per_target(res, separate_scenarios=True)
    if axs is None: axs = utils.Axes(2 + hetro_models*6, 2, axsize)
    a = 0

    nlls, _ = show_res_map(res, 'NLL', ax=axs[a], scenarios=scenarios, models=models)
    a += 1
    rmses, _ = show_res_map(res, 'SE', lambda x: np.sqrt(np.mean(x)), 'RMSE', axs[a], scenarios=scenarios, models=models)
    a += 1
    
    if print_table:
        for tmp_data in (nlls, rmses):
            for row,sc in zip(tmp_data,scenarios):
                i0 = np.argmin(row)
                print(sc, end='')
                for i,el in enumerate(row):
                    print(' & ', end='')
                    if i == i0:
                        print('{{\\bf {:.1f}\}}'.format(el), end='')
                    else:
                        print(f'{el:.1f}', end='')
                print(r' \\')
            print()

    if hetro_models:
        base_models = [m for m in models if not m.startswith('O')]
        base_models_ids = [i for i,m in enumerate(models) if not m.startswith('O')]
        opt_models_ids = [i for i,m in enumerate(models) if m.startswith('O')]
        if base_models_ids and opt_models_ids:
            res_summary = pd.DataFrame(dict(
                scenario = np.repeat(scenarios, len(base_models)),
                model = len(scenarios) * list(base_models),
                base_nll = nlls[:, base_models_ids].reshape(-1),
                opt_nll = nlls[:, opt_models_ids].reshape(-1),
                nll_base_vs_best = (nlls[:, base_models_ids]/np.min(nlls[:, base_models_ids],axis=1)[:,np.newaxis]).reshape(-1),
                nll_opt_vs_base = (nlls[:, opt_models_ids]/nlls[:, base_models_ids]).reshape(-1),
                base_mse = rmses[:, base_models_ids].reshape(-1),
                opt_mse = rmses[:, opt_models_ids].reshape(-1),
                rmse_base_vs_best = (rmses[:, base_models_ids]/np.min(rmses[:, base_models_ids],axis=1)[:,np.newaxis]).reshape(-1),
                rmse_opt_vs_base = (rmses[:, opt_models_ids]/rmses[:, base_models_ids]).reshape(-1),
            ))
            # utils.compare_quantiles(res_summary, 'nll_base_vs_best', 'nll_opt_vs_base', xbins=4, box=False, axs=[axs[a]])
            axs[a].plot(res_summary.nll_base_vs_best.values, 1/res_summary.nll_opt_vs_base.values, 'bo')
            utils.labels(axs[a], 'suboptimality among basic models', 'optimization gain', 'NLL', 15)
            a += 1
            # utils.compare_quantiles(res_summary, 'mse_base_vs_best', 'mse_opt_vs_base', xbins=4, box=False, axs=[axs[a]])
            axs[a].plot(res_summary.rmse_base_vs_best.values, 1/res_summary.rmse_opt_vs_base.values, 'bo')
            utils.labels(axs[a], 'suboptimality among basic models', 'optimization gain', 'RMSE', 15)
            a += 1

            n = len(res_summary)
            y = sorted(1/res_summary.nll_opt_vs_base.values)
            axs[a].axhline(np.mean(y), color='b', linestyle='--')
            axs[a].plot(100*np.arange(n)/(n-1), y, 'b.-')
            utils.labels(axs[a], 'Quantile over scenarios & models [%]', 'NLL(estimated)/NLL(optimized)')
            a += 1
            y = sorted(1/res_summary.rmse_opt_vs_base.values)
            axs[a].axhline(np.mean(y), color='b', linestyle='--')
            axs[a].plot(100*np.arange(n)/(n-1), y, 'b.-')
            utils.labels(axs[a], 'Quantile over scenarios & models [%]', 'RMSE(estimated)/RMSE(optimized)')
            a += 1

        res_summary = pd.DataFrame(dict(
            scenario = np.repeat(scenarios, len(models)),
            model = len(scenarios) * list(models),
            nll = nlls.reshape(-1),
            rmse = rmses.reshape(-1),
            nll_vs_best = (nlls/np.min(nlls[:, base_models_ids],axis=1)[:,np.newaxis]).reshape(-1),
            rmse_vs_best = (rmses/np.min(rmses[:, base_models_ids],axis=1)[:,np.newaxis]).reshape(-1),
        ))
        res_summary['optimized'] = [m.startswith('O') for m in res_summary.model]
        n = len(res_summary) // 2
        y = sorted(res_summary[~res_summary.optimized].nll_vs_best.values)
        axs[a].plot(100*np.arange(n)/(n-1), y, '.-', label=f'Estimated (avg={np.mean(y):.2f})')
        y = sorted(res_summary[ res_summary.optimized].nll_vs_best.values)
        axs[a].plot(100*np.arange(n)/(n-1), y, '.-', label=f'Optimized (avg={np.mean(y):.2f})')
        utils.labels(axs[a], 'Quantile over scenarios & models [%]', 'Relative NLL\n(1 = best estimated model)', fontsize=14)
        axs[a].legend(fontsize=13)
        a += 1
        y = sorted(res_summary[~res_summary.optimized].rmse_vs_best.values)
        axs[a].plot(100*np.arange(n)/(n-1), y, '.-', label=f'Estimated (avg={np.mean(y):.2f})')
        y = sorted(res_summary[ res_summary.optimized].rmse_vs_best.values)
        axs[a].plot(100*np.arange(n)/(n-1), y, '.-', label=f'Optimized (avg={np.mean(y):.2f})')
        utils.labels(axs[a], 'Quantile over scenarios & models [%]', 'Relative RMSE\n(1 = best estimated model)', fontsize=14)
        axs[a].legend(fontsize=13)
        a += 1

    plt.tight_layout()
    if savefig is not None:
        if len(savefig)<5 or savefig[-4]!='.':
            savefig += '.png'
        plt.savefig(savefig, bbox_inches='tight')

    axs2 = utils.Axes(2,1,(12,4.5))
    a = 0
    # sns.set(font_scale=1.3)
    sns.barplot(data=res_tar, x='scenario', hue='model', order=scenarios, hue_order=models, y='NLL', ax=axs2[a], capsize=.07)
    axs2[a].set_xticklabels(axs2[a].get_xticklabels(), fontsize=14, rotation=30)
    axs2[a].legend(fontsize=13)
    ymax = 2*np.max(np.min(nlls,axis=1))
    axs2[a].set_ylim((0,ymax))
    a += 1
    sns.barplot(data=res_tar, x='scenario', hue='model', order=scenarios, hue_order=models, y='SE', ax=axs2[a], capsize=.07)
    axs2[a].set_xticklabels(axs2[a].get_xticklabels(), fontsize=14, rotation=30)
    axs2[a].set_ylabel('MSE')
    axs2[a].legend(fontsize=13)
    ymax = 2*np.max(np.min(rmses**2,axis=1))
    axs2[a].set_ylim((0,ymax))
    a += 1
    # sns.set(font_scale=1)

    plt.tight_layout()
    return axs

def show_res_map(res, score, mean=np.mean, lab=None, ax=None, scenarios=None, models=None, scores=None):
    if ax is None: ax = utils.Axes(1,1,(6,4))[0]
    if lab is None: lab = score
    if scenarios is None: scenarios = res.scenario.unique()
    if models is None: models = res.model.unique()
    # get scores map
    if scores is None:
        scores = np.zeros((len(scenarios), len(models)))
        for i,s in enumerate(scenarios):
            for j,m in enumerate(models):
                scores[i][j] = mean(res[(res.model==m)&(res.scenario==s)][score])
    # normalize rows
    s0 = np.min(scores, axis=1)
    s1 = np.max(scores, axis=1)
    s0 = np.repeat(s0[:,np.newaxis], scores.shape[1], axis=1)
    s1 = np.repeat(s1[:,np.newaxis], scores.shape[1], axis=1)
    rel_scores = - (scores-s0) / (s1-s0)
    # plot
    labs = np.array([[f'{s:.1f}' for s in row] for row in scores])
    sns.heatmap(rel_scores, annot=labs, cmap='RdYlGn', cbar=False, linewidths=0.3, fmt='', ax=ax)
    ax.set_xticklabels(models, rotation=90, fontsize=13)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios, rotation=0, fontsize=13)
    ax.set_title(lab, fontsize=15)
    return scores, ax

def test_analysis(res, axs=None, axargs=None, ref_model=None, scores=('NLL','SE','AE'),
                  by_scenario=False, nll_skip=1, per_target=False):
    models = sorted(np.unique(res.model))
    multi_models = [m for m in models if \
                    len(np.unique(res[res.model==m].predicted_gaussian))>1]
    if ref_model is None: ref_model = models[0]
    if axs is None:
        if axargs is None: axargs = {}
        axargs = utils.update_dict(axargs, dict(axsize=(5,3.5), W=4//(1+(len(multi_models)>4))))
        axs = utils.Axes(len(scores)*4 + 2*len(multi_models), **axargs)
    a = 0

    res0 = res
    if per_target:
        res = res.groupby('target').apply(lambda dd:dd.groupby('model').apply(lambda d:pd.DataFrame(dict(
            scenario=[d.scenario.values[0]], target=[d.target.values[0]], model=[d.model.values[0]], wt=[len(d)],
            NLL=[d.NLL.mean()], SE=[d.SE.mean()], AE=[d.AE.mean()], loss=[d.loss.mean()]
        ))))
        for score in scores:
            res[score] = res[score] * res.wt / res.wt.mean()

    # scores summary
    for score in scores:
        if len(np.unique(res.scenario)) > 1:
            utils.compare_quantiles(res, 'scenario', score, 'model', hbins=20, axs=[axs[a+0]],
                                    mean=(lambda x:np.sqrt(np.mean(x))) if score=='SE' else np.mean)
        else:
            tmp = res.copy()
            labs = []
            for m in models:
                if score == 'SE':
                    avg = np.sqrt(tmp[tmp.model==m][score].mean())
                else:
                    avg = tmp[tmp.model==m][score].mean()
                lab = f'{m} ({avg:.0f})'
                labs.append(lab)
                tmp.loc[tmp.model==m, 'model'] = lab
            sns.boxplot(data=tmp, x='scenario', hue='model', y=score, hue_order=labs, showmeans=True, ax=axs[a+0])
            axs[a+0].set_yscale('log')

        utils.compare_with_ref(res, score, 'model', ref=ref_model, x_ord=models,
                               hue='scenario' if by_scenario else None, ax=axs[a+1], font=16)

        for model in models:
            utils.plot_quantiles(res[score][res.model==model], ax=axs[a+2], plot_mean=True, label=model)
        axs[a+2].set_yscale('log')
        utils.labels(axs[a+2], 'detection quantile', score, fontsize=16)
        axs[a+2].legend()

        utils.compare_quantiles(res0[res0.t>=nll_skip] if score=='NLL' else res0, 't', score, 'model', xbins=8, hbins=20, axs=[axs[a+3]],
                                mean=(lambda x:np.sqrt(np.mean(x))) if score=='SE' else np.mean)
        a += 4

    # distribution over gaussians
    for model in multi_models:
        for gg in ('predicted_gaussian', 'matched_gaussian'):
            d = res0[res0.model==model]
            times, gaussians, counts = [], [], []
            for t in np.unique(d.t):
                tmp_counts_sum = 0
                tmp_counts = []
                for g in np.unique(d[gg]):
                    times.append(t)
                    gaussians.append(f'G{g:d}')
                    count = ((d.t==t) & (d[gg]==g)).sum()
                    tmp_counts.append(count)
                    tmp_counts_sum += count
                counts.extend([100*c/tmp_counts_sum for c in tmp_counts])
            sns.lineplot(data=pd.DataFrame({'t':times,gg:gaussians,'count [%]':counts}),
                         x='t', hue=gg, y='count [%]', markers=True, ax=axs[a])
            axs[a].set_title(model, fontsize=16)
            a += 1

    plt.tight_layout()
    return axs

def diagnose_target(model, X, Y, xrange=200, delta=10, plot_with_priors=True):
    T = len(X)
    axs = utils.Axes(2*T + 2, 4, axsize=(5,5), fontsize=16)
    a = 0

    # model performance
    X, Y, dts = extract_times(X, Y)
    model.init_state()
    preds = []
    ests = []
    for t,(x,y,dt) in enumerate(zip(X,Y,dts)):
        model.predict(dt=dt)
        # print(model.base_g, model.x)
        preds.append(model.get_pos())
        pred_loss, nll, g1 = model.loss_pred(y, detailed=True)
        nll = nll.item()
        pred_loss = pred_loss.item()
        show_nll_map(model, y, X[t-1] if t>0 else np.zeros(4), xrange=xrange, delta=delta,
                     use_priors=plot_with_priors, ax=axs[a], tit=f'[{t:d}, pred] NLL={nll:.0f}')
        a += 1

        model.do_update(x)
        # print('\t', model.base_g, model.x)
        ests.append(model.get_pos())
        update_loss, mse, g2 = model.loss_update(y, detailed=True)
        mse = mse.item()
        update_loss = update_loss.item()
        show_nll_map(model, y, x, xrange=xrange, delta=delta,
                     use_priors=plot_with_priors, ax=axs[a], tit=f'[{t:d}, update] |ERR|={np.sqrt(mse):.0f}')
        a += 1

    # episode info
    axs[a].plot(Y[0,0], Y[0,1], 'k>', markersize=20)
    axs[a].plot(Y[:,0], Y[:,1], 'go-', markersize=8, label='target')
    axs[a].plot(X[:,0], X[:,1], 'yo', markersize=6, label='observation')
    axs[a].plot([p[0] for p in preds], [p[1] for p in preds], 'bo', markersize=6, label='predition')
    axs[a].plot([p[0] for p in ests], [p[1] for p in ests], 'ro', markersize=6, label='estimation')
    axs.labs(a, 'x', 'y', f'target position ({T:d} steps)')
    axs[a].legend(fontsize=13)
    a += 1

    # total errors
    axs[a].plot(np.arange(T), [np.linalg.norm(x) for x in (np.array(preds)-Y)], 'b.-', label='prediction')
    axs[a].plot(np.arange(T), [np.linalg.norm(x) for x in (np.array(ests) -Y)], 'r.-', label='estimation')
    axs.labs(a, 't', '|error| [m]')
    axs[a].legend(fontsize=13)

    plt.tight_layout()

    return axs

def show_nll_map(model, y, z, xrange=500, delta=10, use_priors=True, tit='NLL', ax=None):
    if ax is None:
        ax = utils.Axes(1, 1, axsize=(6, 6))[0]

    hm = np.zeros(((2 * xrange) // delta + 1, (2 * xrange) // delta + 1))
    n = len(hm)
    best = (np.inf, 0, 0)
    for i, dy in enumerate(range(-xrange, xrange + 1, delta)):
        for j, dx in enumerate(range(-xrange, xrange + 1, delta)):
            hm[i, j] = model.calc_nll((z[0] + dx, z[1] + dy, z[2], z[3]), prior=use_priors)
            if hm[i, j] < best[0]:
                best = (hm[i, j], n-1-i, j)
    hm = hm[::-1, :]

    sns.heatmap(hm, ax=ax)
    ax.set_xticklabels([int(z[0] - xrange + delta * int(l.get_text())) for l in ax.get_xticklabels()], rotation=90)
    ax.set_yticklabels([int(z[1] - xrange + delta * int(l.get_text())) for l in ax.get_yticklabels()][::-1], rotation=0)

    # observation
    ax.plot(n//2+0.5, n//2+0.5, 'yo', markersize=16)
    # estimation
    ax.plot(best[2]+0.5, best[1]+0.5, 'ro', markersize=12)
    v = model.x[model.base_g].detach().cpu().numpy()[3:5]
    v = v*n/2 / 200 # 200 m/s = full screen
    ax.arrow(best[2]+0.5, best[1]+0.5, v[0], -v[1], head_width=len(hm)/30, head_length=len(hm)/30, fc='r', ec='r')
    # true state
    tmp = np.clip((y[:2]-z[:2]) / delta, -n/2, n/2)
    ax.plot(n//2+0.5+tmp[0], n//2+0.5-tmp[1], 'go', markersize=12)
    v = y[3:5]
    v = v*n/2 / 200 # 200 m/s = full screen
    ax.arrow(n//2+0.5+tmp[0], n//2+0.5+tmp[1], v[0], -v[1], head_width=len(hm)/30, head_length=len(hm)/30, fc='g', ec='g')

    plab = '(' + ', '.join([f'{x:.1f}' for x in model.p.detach().cpu().numpy()]) + ')'
    ylab = '(' + ', '.join([f'{x:.0f}' for x in y[:3]]) + ')'
    zlab = '(' + ', '.join([f'{x:.0f}' for x in z[:3]]) + ')'
    xlabs = ['(' + ','.join([f'{x:.0f}' for x in xx.detach().cpu().numpy()[:3]]) + ')' for xx in model.x]
    xlab = ', '.join(xlabs)
    vlabs = ['(' + ','.join([f'{x:.0f}' for x in xx.detach().cpu().numpy()[3:]]) + ')' for xx in model.x]
    vlab = ', '.join(vlabs)
    alabs = ['(' + ','.join([f'{x:.0f}' for x in xx.detach().cpu().numpy()]) + ')' for xx in model.a]
    alab = ', '.join(alabs)
    Plab = '(' + ', '.join([f'{x:.0f}' for x in np.diag(model.P.detach().cpu().numpy())][:3]) + ')'

    utils.labels(ax, 'x', 'y',
                 f'{tit}\np = {plab}\n$x_{{true}}$ = {ylab}, $x_{{obs}}$ = {zlab}\n$x_{{est}}$ = {xlab}\n$v_{{est}}$ = {vlab}\n$a_{{est}}$ = {alab}\nP = {Plab}',
                 fontsize=11)

def show_weights(model, g=0, layer=0, axs=None):
    if axs is None: axs = utils.Axes(2,1,axsize=(16,3.5))

    x = model.layers[g][layer].weight_ih.detach().cpu().numpy()

    sns.heatmap(x.transpose(), ax=axs[0])

    axs[1].bar(np.arange(x.shape[1]), np.sum(np.abs(x), axis=0))
    utils.labels(axs[1], 'input element', 'total weight', fontsize=14)

    plt.tight_layout()
    return axs

def multi_scenario_models_plot(scenarios, models_args, i, **kwargs):
    if isinstance(i, int): i = len(scenarios) * [i]
    axs = []
    for count, title in enumerate(scenarios):
        # load scenario
        X, Y, tars = load_data(fname=f'{title}_test')
        ii = i[count]
        X, Y = X[ii], Y[ii]
        # load models
        models = [NT.NeuralKF(**a) for a in models_args]
        for m in models:
            m.load_model(f'{title}_{m.title}')
        # run models
        axs.append(show_models_on_target(X, Y, models, axs=None, **kwargs))
    return axs

def show_models_on_target(X, Y, models, axs=None, dimx=0, dimy=1):
    if axs is None: axs = utils.Axes(2,2,(8,6))
    model_names = [m.title for m in models]
    predictions = {}
    estimations = {}

    X, Y, dts = extract_times(X, Y)

    axs[0].plot(Y[:,dimx], Y[:,dimy], 'k.-', label='target')
    axs[0].plot(Y[:1,dimx], Y[:1,dimy], 'k>', markersize=8)
    axs[0].plot(Y[-1:,dimx], Y[-1:,dimy], 'ks', markersize=8)
    axs[1].plot(Y[:,dimx], Y[:,dimy], 'k.-', label='target')
    axs[1].plot(Y[:1,dimx], Y[:1,dimy], 'k>', markersize=8)
    axs[1].plot(Y[-1:,dimx], Y[-1:,dimy], 'ks', markersize=8)
    axs[1].plot(X[:,dimx], X[:,dimy], 'kx', label='observation')
    for model, nm in zip(models, model_names):
        model.init_state()
        preds = []
        ests = []
        for t,(x,y,dt) in enumerate(zip(X,Y,dts)):
            model.predict(dt=dt)
            preds.append(model.get_pos())
            # pred_loss, nll, g1 = model.loss_pred(y, detailed=True)
            # nll = nll.item()
            # pred_loss = pred_loss.item()
            model.do_update(x)
            ests.append(model.get_pos())
            # update_loss, mse, g2 = model.loss_update(y, detailed=True)
            # mse = mse.item()
            # update_loss = update_loss.item()
            # show_nll_map(model, y, x, xrange=xrange, delta=delta,
            #              use_priors=plot_with_priors, ax=axs[a], tit=f'[{t:d}, update] |ERR|={np.sqrt(mse):.0f}')
            # a += 1
        predictions[nm] = preds
        estimations[nm] = ests

        # plot
        x = [p[dimx] for p in preds[1:]]
        y = [p[dimy] for p in preds[1:]]
        axs[0].plot(x, y, '.-', label=nm)
        x = [p[dimx] for p in ests]
        y = [p[dimy] for p in ests]
        axs[1].plot(x, y, '.-', label=nm)

    utils.labels(axs[0], ['x','y','z'][dimx], ['x','y','z'][dimy], 'Predictions', 15)
    utils.labels(axs[1], ['x','y','z'][dimx], ['x','y','z'][dimy], 'Estimations', 15)
    axs[0].legend(fontsize=12)
    axs[1].legend(fontsize=12)

    plt.tight_layout()

    return axs

def compare_noise_models(scenarios, models_args, noise='R', log=False, polar_labs=None, fontsize=None):
    axs = utils.Axes(len(scenarios)*(len(models_args)+1), len(models_args)+1)
    a = 0

    for title in scenarios:
        # load models
        models = [NT.NeuralKF(**a) for a in models_args]
        for m in models:
            m.load_model(f'{title}_{m.title}')
        # show noise model
        for m in models:
            R = m.get_cov_matrix(noise,0).detach().numpy()
            h = sns.heatmap(np.log(R) if log else R, ax=axs[a], cmap='Reds', annot=R, fmt='.0f',
                            annot_kws=None if fontsize is None else dict(size=fontsize))
            h.xaxis.set_ticks_position("top")
            axs[a].set_title(f'[{title}] {m.title} / {noise}', fontsize=16)
            if noise == 'Q':
                axs[a].set_xticklabels(['x','y','z','vx','vy','vz'], fontsize=15)
                axs[a].set_yticklabels(['x','y','z','vx','vy','vz'], fontsize=15)
            else:
                polar = 'p' in m.title if polar_labs is None else polar_labs
                if polar:
                    axs[a].set_xticklabels(['r','theta','phi','Dop'], fontsize=15)
                    axs[a].set_yticklabels(['r','theta','phi','Dop'], fontsize=15)
                else:
                    axs[a].set_xticklabels(['x','y','z','Dop'], fontsize=15)
                    axs[a].set_yticklabels(['x','y','z','Dop'], fontsize=15)
            a += 1
        # compare models
        for m in models:
            R = m.get_cov_matrix(noise,0).diag().detach().numpy()
            R = R / R.mean()
            axs[a].plot(R, label=m.title)
        axs[a].set_ylabel(f'normalized diag\n(relative variance)', fontsize=14)
        axs[a].set_title(f'{title}', fontsize=14)
        axs[a].set_xticks(np.arange(4 if noise=='R' else 6))
        axs[a].set_xticklabels(('x/R','y/theta','z/phi','dop') if noise=='R' else ('x','y','z','vx','vy','vz'), fontsize=12)
        axs[a].legend(fontsize=12)
        a += 1

    plt.tight_layout()
    return axs
