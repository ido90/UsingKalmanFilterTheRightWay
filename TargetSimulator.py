'''
Given a scenario, multiple iid episodes can be randomly generated.
Every episode includes multiple targets.
Every target has a trajectory with multiple flight phases (currently either straight-line or turn).
Every phase is built according to randomly-generated defining arguments.

simulate_episodes() generates multiple episodes for a scenario, according to the params specified below.
main() wraps simulate_episodes() with few pre-defined sets of params.


Output of simulate_episodes():
    dd:             data frame of meta-data about the trajectories. every row is one target.
    targets:        targets[episode][target] contains the trajectory data as a data-frame.
                    columns "x_*" are state variables.
                    "phase" is the index of trajectory phase (phases should alternate between straight lines and turns).
    target_args:    target_args[episode][target] contains the random parameters that were drawn for the target and determined its trajectory.
                    it mostly contains initial state params, number of turns, and params for every turn and straight interval.

Input for simulate_episodes() (see main() for examples):
    n_episodes (2):     number of episodes.
    dt (1):             temporal resolution.
    acc (40):           target acceleration (in the current implementation, at any time the acceleration is either 0 or acc).
    n_targets_args (2): number of targets per episode / dict of args for draw_int().
    -----------------------
    init_args (None):   dict of args for initialize_state(t0, dt, X0, dx, V0, dV, v_radial=True).
                        V0,dV are interpreted in polar units if v_radial=True, otherwise cartesian.
    n_turns_args (1):   number of turns per target / dict of args for draw_int(n1, n2=None, method='unif').
                        the trajectory will consist of n+1 lines and n turns between them.
                        each line/turn is randomly generated according to the generation-args below.
    line_args (None):   dict of p_acc=P(accelerate) (otherwise const speed); t_mean=E[time duration]; t_sigma=std(time duration).
                        statistics (mean,std) are somewhat skewed due to disturbing implementation of lognormal distribution.
    turn_args (None):   dict of p_left=P(left turn); a_mean=E[turn angle]; a_sigma=std(turn angle).
    -----------------------
    seeds (None):       list of seeds for episodes. default is (0,...,n_episodes-1).
    title ('scenario'): name of scenario.
    do_save (False):    whether to save results. if string, interpreted as True + replacing title as filename.


Currently unsupported:
- heterogeneous targets in episode
- vertical turns (vz currently comes from random initial vz0 + straight-forward acceleration)
- max speed


Module structure:
    MAIN
    GENERAL FUNCTIONS
    TRAJECTORIES INTERVALS GENERATORS
    FULL TRAJECTORIES
    EPISODES GENERATOR
    VISUALIZATION TOOLS

_______________________

Written by Ido Greenberg, 2020
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

import sys
from pathlib import Path
import pickle as pkl
from warnings import warn

import utils
import BasicTargetSimulator as BTG


##################   MAIN   ##################

DEFAULT_INIT_TARGET_ARGS = dict(t0=10, dt=2, X0=(0,0,0), dx=100, V0=(100,45,80), dV=(20,30,5))
DEFAULT_LINE_ARGS = dict(p_acc=0.5, t_mean=10, t_sigma=3, vmax0=150)
DEFAULT_TURN_ARGS = dict(p_left=0.5, a_mean=45, a_sigma=10, p_vertical=0.3, vertical_fac=1/5)

def single_line_hyperparams(acc=40, p_acc=1, t_mean=12, t_sigma=1.5):
    init_args = dict(V0=(100,45,90), dV=(0,0,0))
    n_turns = 0
    line_args = dict(p_acc=p_acc, t_mean=t_mean, t_sigma=t_sigma)
    turn_args = None
    return dict(acc=acc, init_args=init_args, n_turns_args=n_turns, line_args=line_args, turn_args=turn_args)

def main(mode=0, do_plot=2):
    if mode == 0:
        target_hyper_args = single_line_hyperparams()
        dd, targets, target_args = simulate_episodes(
            n_episodes=2, dt=1, n_targets_args=2, seeds=None, title='line', do_save=False, **target_hyper_args)
    elif mode == 1:
        target_hyper_args = single_line_hyperparams()
        dd, targets, target_args = simulate_episodes(
            n_episodes=10, dt=1, n_targets_args=10, seeds=None, title='line', do_save=False, **target_hyper_args)
    elif mode == 2:
        dd, targets, target_args = simulate_episodes(
            n_episodes=10, dt=1, n_targets_args=10, seeds=None, title='few_turns', do_save=False,
            n_turns_args=dict(n1=2,n2=2,method='lognormal'),
            line_args = dict(p_acc=0.3, t_mean=6, t_sigma=2))
    else:
        raise ValueError(mode)

    if do_plot >= 1:
        show_episodes(targets, target_args)
        if do_plot >= 2:
            show_episodes_3D(targets, target_args)
            if do_plot >= 3:
                show_episodes_3D(targets, target_args, velocity=True)

    return dd, targets, target_args

def load_scenario(nm, base_path=Path('Scenarios/')):
    if not nm.endswith('.pkl'): nm += '.pkl'
    with open(base_path / nm, 'rb') as fd:
        dd, targets, target_args = pkl.load(fd)
    return dd, targets, target_args

##################   GENERAL FUNCTIONS   ##################

def cart2polar(x, y, z):
    r = np.linalg.norm((x,y,z))
    theta = np.arctan2(x,y)
    phi = np.arccos(z/r)
    return r, theta, phi

def polar2cart(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def lognormal(mu, sigma):
    if sigma == 0: return mu
    return np.random.lognormal(np.log(mu ** 2 / np.sqrt(mu ** 2 + sigma ** 2)),
                               np.sqrt(np.log(1 + sigma ** 2 / mu ** 2)) )

def set_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)

def draw_seed():
    return np.random.randint(2**31)

##################   TRAJECTORIES INTERVALS GENERATORS   ##################

def line(t0, dt, X0, acc, duration, vmax, theta=None, phi=None):
    # input pre-processing
    Vabs = np.linalg.norm(X0[3:])
    if phi is None: phi = np.arccos(X0[5]/Vabs)
    if theta is None: theta = np.arctan2(X0[4],X0[3])

    # generate trajectory
    times = np.arange(t0, t0 + duration, dt)
    X, V = BTG.line(times, X0[:3], X0[3:], acc, vmax=vmax, theta=theta, phi=phi)

    # post-processing
    df = pd.DataFrame(
        np.vstack((times, X, V)).transpose(),
        columns=('t', 'x_x', 'x_y', 'x_z', 'x_vx', 'x_vy', 'x_vz'),
    )
    return df

def circle(t0, dt, X0, acc=None, duration=None, angle=None, right=True, vertical=False, max_rise=30,
           short_phase_warning=False):
    # input pre-processing
    Vabs = np.linalg.norm(X0[3:6])
    if angle is not None:
        angle += 360
        angle = angle % 720
        angle -= 360
        right = angle >= 0
    if acc is None:
        distance = Vabs * duration
        radius = distance / (2*np.pi) * 360 / np.abs(angle)
        acc = Vabs**2/radius
    elif duration is None:
        radius = Vabs**2 / acc
        distance = 2*np.pi * radius * np.abs(angle) / 360
        duration = distance / Vabs + 1
    if short_phase_warning and duration < short_phase_warning:
        warn(f'Short turn: theta,v,a,t = {angle}, {Vabs}, {acc}, {duration}')

    # generate trajectory
    times = np.arange(t0, t0+duration, dt)
    if vertical:
        X, V = BTG.vert_circle(times, acc, X0[:3], X0[3:6], right, phi_max=max_rise)
    else:
        X, V = BTG.hor_circle(times, acc, X0[:3], X0[3:6], right)

    # post-processing
    df = pd.DataFrame(
        np.vstack((times, X, V)).transpose(),
        columns=('t', 'x_x', 'x_y', 'x_z', 'x_vx', 'x_vy', 'x_vz'),
    )
    return df

##################   FULL TRAJECTORIES   ##################

def simulate_target(dt=1, acc=40, init_args=None, n_turns_args=1, line_args=None, turn_args=None,
                    interval_probs=None, old_mode=True, center=False, seed=None):
    if isinstance(acc, (list,tuple)):
        acc = np.random.choice(acc)
    args = generate_trajectory_params(max_acc=acc, init_args=init_args, n_turns_args=n_turns_args,
                                      line_args=line_args, turn_args=turn_args, interval_probs=interval_probs,
                                      old=old_mode, seed=seed)
    args['acc'] = acc
    args['t0'] = dt*int(np.round(args['t0']/dt))
    df = build_trajectory(args['t0'], dt, args['X0'], args['intervals'], center=center)
    args['tf'] = df.t.values[-1]
    args['T'] = args['tf'] - args['t0']

    return df, args

def generate_interval(max_acc, vmax, line_args, turn_args, interval_probs=(0.15,0.25,0.25,0.25,0.05,0.05)):
    '''
    A single-interval generator, interfacing to the new-mode simulation (the one with the non-patterned intervals)
    '''
    interval_type = np.random.choice(['constant','acc','left','right','up','down'], p=interval_probs)
    acc = max_acc * (0.5*np.random.random()+0.5) if interval_type!='constant' else 0

    if interval_type in ('constant','acc'):
        duration = 6 + 14*np.random.random() # max(4, (400 + 1200*np.random.random()) / vmax)
        return (
            line,
            dict(
                acc      = acc,
                duration = duration,
                vmax     = vmax
            )
        )
    else:
        vertical = interval_type in ('up','down')
        if vertical:
            ang_params = turn_args['a_mean']*turn_args['vertical_fac'], turn_args['a_sigma']*turn_args['vertical_fac']
        else:
            ang_params = turn_args['a_mean'], turn_args['a_sigma']
        return (
            circle,
            dict(
                acc      = acc/5 if vertical else acc,
                angle    = (1 if interval_type in ('up','right') else -1) * \
                           ((5 if vertical else 30) + (15 if vertical else 300)*np.random.random()), # lognormal(*ang_params),
                vertical = vertical
            )
        )

def generate_trajectory_params(max_acc, init_args=None, n_turns_args=1, line_args=None, turn_args=None,
                               interval_probs=None, seed=None, old=True):
    if init_args is None: init_args = {}
    if isinstance(n_turns_args, int): n_turns_args = dict(n1=n_turns_args)
    if line_args is None: line_args = {}
    if turn_args is None: turn_args = {}

    set_seed(seed)

    init_args = utils.update_dict(
        init_args, DEFAULT_INIT_TARGET_ARGS, force=False)
    line_args = utils.update_dict(
        line_args, DEFAULT_LINE_ARGS, force=False)
    turn_args = utils.update_dict(
        turn_args, DEFAULT_TURN_ARGS, force=False)
    vmax = line_args['vmax0']

    args = {}
    args['seed'] = seed
    args['t0'], args['X0'] = initialize_state(**init_args)
    args['n_turns'] = draw_int(**n_turns_args)

    if not old:
        args['intervals'] = [generate_interval(max_acc, vmax, line_args, turn_args, interval_probs) for _ in range(args['n_turns'])]
        return args

    else:
        args['intervals'] = []
        acc = (np.random.random() < line_args['p_acc'])
        args['intervals'].append(
            (
                line,
                dict(
                    acc      = max_acc * (0.5*np.random.random()+0.5) * acc,
                    duration = max(5, lognormal(line_args['t_mean_acc'] if acc else line_args['t_mean'],
                                                line_args['t_sigma_acc'] if acc else line_args['t_sigma'])),
                    vmax     = vmax
                )
            )
        )
        for i in range(args['n_turns']):
            vertical = np.random.rand() < turn_args['p_vertical']
            if vertical:
                ang_params = turn_args['a_mean']*turn_args['vertical_fac'], turn_args['a_sigma']*turn_args['vertical_fac']
            else:
                ang_params = turn_args['a_mean'], turn_args['a_sigma']
            args['intervals'].append(
                (
                    circle,
                    dict(
                        acc      = max_acc*(0.5*np.random.random()+0.5) / (5 if vertical else 1),
                        angle    =(2*((np.random.random()>turn_args['p_left']))-1) * draw_angle(*ang_params),
                        vertical = vertical
                    )
                )
            )
            acc = (np.random.random()<line_args['p_acc'])
            args['intervals'].append(
                (
                    line,
                    dict(
                        acc      = max_acc * (0.5*np.random.random()+0.5) * acc,
                        duration = max(5, lognormal(line_args['t_mean_acc'] if acc else line_args['t_mean'],
                                                    line_args['t_sigma_acc'] if acc else line_args['t_mean'])),
                        vmax     = vmax
                    )
                )
            )

        return args

def draw_angle(a1, a2, mode='unif'):
    if mode == 'lognormal':
        return lognormal(a1, a2)
    elif mode == 'unif':
        return a1 + (a2-a1) * np.random.random()
    raise ValueError(mode)

def initialize_state(t0, dt, X0, V0, dx=None, dV=None, v_radial=True, unif_x=True, unif_v=False):
    if v_radial:
        V0 = (V0[0], V0[1]*np.pi/180, V0[2]*np.pi/180)
        dV = (dV[0], dV[1]*np.pi/180, dV[2]*np.pi/180)
    t = lognormal(t0, dt)
    X = draw_3D(X0, dx, radial=False, unif=unif_x)
    V = draw_3D(V0, dV, radial=v_radial, pos_rad=True, unif=unif_v, rmin=0.5)
    return t, np.concatenate((X,V))

def draw_3D(X0, dX=None, radial=True, pos_rad=False, unif=False, rmin=0.):
    if unif:
        if radial:
            r = X0 * (rmin**3+(1-rmin**3)*np.random.random()) ** (1 / 3)
            theta = 2 * np.pi * np.random.random()
            phi = np.arccos(1 - 2 * np.random.random())
            X = np.array(polar2cart(r, theta, phi))
        else:
            X = np.array(X0) + np.array(dX) * (2*np.random.random(3)-1)
    else:
        if not radial:
            X = np.array(X0) + np.random.normal(0, dX, 3)
        else:
            r = np.random.normal(X0[0], dX[0])
            if pos_rad and r<0:
                r = np.abs(r)
            theta = np.random.normal(X0[1], dX[1]) % (2*np.pi)
            phi = np.random.normal(X0[2], dX[2]) % np.pi
            X = np.array(polar2cart(r, theta, phi))
    return X

def draw_int(n1, n2=None, method='unif'):
    if n2 is None or n1 == n2:
        return n1
    if method == 'unif':
        return int(np.random.randint(low=n1, high=n2))
    if method == 'lognormal':
        return int(lognormal(mu=n1, sigma=n2))
    raise ValueError(method)

def build_trajectory(t0, dt, X0, intervals, center=False):
    '''
    :param t0: initial time
    :param dt: simulation temporal resolution
    :param X0: initial position & velocity
    :param trajs: list of lists (fun, args)
    :return: df: t|x,y,z,vx,vy,vz

    note - interval creator fun input: t0, dt, X0, more args.
    '''

    # first interval
    fun, args = intervals[0]
    data = fun(t0, dt, X0, **args)
    data['phase'] = 0
    t = data.t.values[-1]
    X = data.iloc[-1,1:]

    # all intervals
    for i, (fun, args) in enumerate(intervals[1:]):
        df = fun(t, dt, X, **args)
        df['phase'] = i+1
        data = pd.concat((data, df[1:]))
        t = data.t.values[-1]
        X = data.iloc[-1, 1:]

    # shift whole trajectory
    if center:
        data['x_x'] += X0[0] - data['x_x'].mean()
        data['x_y'] += X0[1] - data['x_y'].mean()
        data['x_z'] += X0[2] - data['x_z'].mean()

    data.reset_index(drop=True, inplace=True)
    data.set_index(data.t, inplace=True)

    return data

##################   EPISODES GENERATOR   ##################

def simulate_episode(dt=1, acc=40, n_targets_args=2, old_mode=True, interval_probs=None,
                      init_args=None, n_turns_args=1, line_args=None, turn_args=None, center=False, seed=None):
    if isinstance(n_targets_args, int): n_targets_args = dict(n1=n_targets_args)
    set_seed(seed)

    n_targets = draw_int(**n_targets_args)
    targets = []
    target_args = []
    for i in range(n_targets):
        seed = draw_seed()
        df, args = simulate_target(dt=dt, acc=acc, init_args=init_args, n_turns_args=n_turns_args,
                                   line_args=line_args, turn_args=turn_args, interval_probs=interval_probs,
                                   old_mode=old_mode, center=center, seed=seed)
        targets.append(df)
        target_args.append(args)

    return targets, target_args

def simulate_episodes(n_episodes=2, dt=1, acc=40, n_targets_args=2, old_mode=True, interval_probs=None,
                      init_args=None, n_turns_args=1, line_args=None, turn_args=None, center=False,
                      seeds=None, title='scenario', do_save=False, base_path=Path('Scenarios/')):
    if seeds is None: seeds = np.arange(n_episodes)
    if isinstance(seeds, int):
        set_seed(seeds)
        seeds = [draw_seed() for _ in range(n_episodes)]

    dd = pd.DataFrame()
    targets = []
    target_args = []
    for i in range(n_episodes):
        targets_i, target_args_i = simulate_episode(
            dt=dt, acc=acc, n_targets_args=n_targets_args, seed=seeds[i], old_mode=old_mode,
            init_args=init_args, n_turns_args=n_turns_args, line_args=line_args, turn_args=turn_args,
            interval_probs=interval_probs, center=center)
        n_targets = len(targets_i)

        targets.append(targets_i)
        target_args.append(target_args_i)

        tmp = pd.DataFrame(dict(
            scenario = n_targets * [title],
            episode = n_targets * [i],
            target = np.arange(n_targets),
            target_class = [f'{a["acc"]//2:02d}<acc<{a["acc"]:02d}' for a in target_args_i],
            seed_ep = n_targets * [seeds[i]],
            seed_target = [a['seed'] for a in target_args_i],
            t0 = [a['t0'] for a in target_args_i],
            tf = [a['tf'] for a in target_args_i],
            T = [a['T'] for a in target_args_i],
            group = n_targets * [0],
        ))
        dd = pd.concat((dd, tmp))

    if do_save:
        nm = do_save if isinstance(do_save,str) else title
        if not nm.endswith('.pkl'): nm += '.pkl'
        with open(base_path/nm, 'wb') as fd:
            pkl.dump((dd, targets, target_args), fd)

    return dd, targets, target_args

def meta_targets2episodes(dd):
    out = pd.DataFrame()
    count = 0
    for s in np.unique(dd.scenario):
        for ep in np.unique(dd[dd.scenario == s].episode):
            d = dd[(dd.scenario == s) & (dd.episode == ep)]
            d = pd.DataFrame(dict(
                scenario=s,
                episode=ep,
                n_targets=len(d),
                seed=d.seed_ep.values[0],
                Ti=d.t0.min(),
                Tf=(d.t0 + d['T']).max(),
                T=(d.t0 + d['T']).max() - d.t0.min(),
                group=0
            ), index=[count])
            out = pd.concat((out, d))
            count += 1
    out.reset_index(drop=True, inplace=True)
    return out

##################   VISUALIZATION TOOLS   ##################

def show_target(tt, ax=None, cols=None, scope='pos', title='target trajectory'):
    if ax is None: ax = utils.Axes(1)[0]
    if cols is None:
        if scope == 'pos':
            cols = ('x_x','x_y','x_z')
        elif scope == 'velocity':
            cols = ('x_vx', 'x_vy', 'x_vz')
        elif scope == 'all':
            cols = [c for c in tt.columns if c.startswith('x_')]
        else:
            raise ValueError(scope)

    phase_change = np.where(np.diff(tt.phase.values) > 0)[0]
    for t in phase_change:
        ax.axvline(t, color='k', linestyle=':')
    for c in cols:
        ax.plot(np.arange(len(tt)), tt[c], '.-', linewidth=1.3, markersize=5, label=c[2:])
    utils.labels(ax, 't', None, title, 16)
    ax.legend(fontsize=10)

    return ax

def show_episodes(targets, target_args, max_episodes=6, max_targets=6, show_v=False,
                  detailed_lab=True, axs=None, axs_args=None):
    n_episodes = min(max_episodes, len(targets))
    if axs is None:
        if axs_args is None:
            axs_args = {}
        axs_args = utils.update_dict(axs_args, dict(W=4))
        axs_args = utils.update_dict(axs_args, dict(axsize=(16/axs_args['W'],12/axs_args['W'])))
        axs_args['N'] = (1+show_v)*n_episodes
        axs = utils.Axes(**axs_args)

    for ep in range(n_episodes):
        n_targets = min(max_targets, len(targets[ep]))
        ax0 = axs[(1+show_v)*ep]
        if show_v:
            ax1 = axs[(1+show_v)*ep+1]
        for i in range(n_targets):
            lab = f'{i} (acc<{target_args[ep][i]["acc"]})' if detailed_lab else f'{i}'
            x = targets[ep][i].x_x.values
            y = targets[ep][i].x_y.values
            h = ax0.plot(x, y, '.-', linewidth=1, markersize=4, label=lab)[0]
            c = h.get_color()
            ax0.plot(x[0], y[0], '>', color=c, markersize=8)
            ax0.plot(x[-1], y[-1], 's', color=c, markersize=8)

            if show_v:
                vx = targets[ep][i].x_vx.values
                vy = targets[ep][i].x_vy.values
                ax1.plot(vx, vy, '.-', color=c, linewidth=1, markersize=4, label=str(i))
                ax1.plot(vx[0], vy[0], '>', color=c, markersize=8)
                ax1.plot(vx[-1], vy[-1], 's', color=c, markersize=8)

        ti = np.min([a["t0"] for a in target_args[ep]])
        tf = np.max([a["tf"] for a in target_args[ep]])
        utils.labels(ax0, 'x', 'y',
                     f'[{ep:d}/{len(targets)}] {len(targets[ep]):d} targets, {ti:.0f}<=t<={tf:.0f}', 13)
        if show_v: utils.labels(ax1, 'vx', 'vy', f'[{ep:d}/{len(targets)}]', 13)
        ax0.legend(fontsize=10)
        if show_v: ax1.legend(fontsize=10)

    plt.tight_layout()
    return axs

def show_episodes_3D(targets, target_args, max_episodes=3, max_targets=5, velocity=False, tit='', figsize=(6,4.5), axs=None):
    n_episodes = min(max_episodes, len(targets))
    if axs is None: axs = []
    if tit: tit = f' ({tit})'

    for ep in range(n_episodes):
        n_targets = min(max_targets, len(targets[ep]))
        plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        axs.append(ax)
        for i in range(n_targets):
            x = targets[ep][i][('x_x','x_vx')[velocity]].values
            y = targets[ep][i][('x_y','x_vy')[velocity]].values
            z = targets[ep][i][('x_z','x_vz')[velocity]].values
            h = ax.plot3D(x, y, z, '.-', linewidth=1, markersize=3.5, label=str(i))[0]
            c = h.get_color()
            ax.plot([x[0]], [y[0]], [z[0]], '>', color=c, markersize=8)
            ax.plot([x[-1]], [y[-1]], [z[-1]], 's', color=c, markersize=8)

        ti = np.min([a["t0"] for a in target_args[ep]])
        tf = np.max([a["tf"] for a in target_args[ep]])
        utils.labels(ax, ('x','vx')[velocity], ('y','vy')[velocity],
                     f'[{ep:d}/{len(targets)}] {len(targets[ep]):d} targets, {ti:.0f}<=t<={tf:.0f}'+tit, 13)
        ax.legend(fontsize=11)

    return axs

def scenarios_summary(dd, axs=None):
    if axs is None:
        axs = utils.Axes(3, W=3, axsize=(5,3.5))
    a = 0

    # n_episodes
    episodes_per_scenario = dd.groupby('scenario').apply(lambda d: len(np.unique(d.episode)))
    axs[a].bar(np.unique(dd.scenario), episodes_per_scenario)
    utils.labels(axs[a], 'scenario', 'number of episodes', fontsize=16)
    a += 1

    # n_targets
    scenarios = np.unique(dd.scenario)
    mm = dd.groupby('scenario').apply(lambda d: pd.DataFrame(dict(
        scenario = len(np.unique(d.episode)) * [d.scenario.values[0]],
        n_targets = [(d.episode==ep).sum() for ep in np.unique(d.episode)],
        episode_len = [(d[d.episode==ep].t0+d[d.episode==ep]['T']).max()-d[d.episode==ep].t0.min() for ep in np.unique(d.episode)],
    )))
    mm = pd.concat([mm.loc[s] for s in scenarios])
    sns.boxplot(data=mm, x='scenario', y='n_targets', showmeans=True, ax=axs[a])
    utils.labels(axs[a], 'scenario', 'targets per episode', None, 16)
    a += 1

    sns.boxplot(data=mm, x='scenario', y='episode_len', showmeans=True, ax=axs[a])
    utils.labels(axs[a], 'scenario', 'episode length [s]', None, 16)
    a += 1

    plt.tight_layout()
    return axs
