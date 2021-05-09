import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import pickle as pkl
from collections import Counter

import utils
import TargetSimulator as Sim
import SensorGenerator as Rad
import PredictionLab as PLAB

###########   SCENARIOS META-ARGUMENTS   ###########

SCENARIOS_META_ARGS = dict(
    cart_iso_cent = dict(
        accs_train = (0,),
        accs_test = (0,),
        n_train = 1500,
        n_test = 50,
        n_targets_test = 20,
        old_mode = True,
        n_turns = 0,
        t_mean = 30,
        t_sigma = 8,
        polar = False,
        unif_x = False,
        v_radial = False,
        V0 = (0,0,0),
        dx = (200,200,200),
        dV = (80,80,80),
        center = True,
    ),
    # cart_cent = dict(
    #     accs_train = (0,),
    #     accs_test = (0,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = True,
    #     n_turns = 0,
    #     t_mean = 30,
    #     t_sigma = 8,
    #     polar = False,
    #     unif_x = False,
    #     dx = (200,200,200),
    #     center = True,
    # ),
    # iso_cent = dict(
    #     accs_train = (0,),
    #     accs_test = (0,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = True,
    #     n_turns = 0,
    #     t_mean = 30,
    #     t_sigma = 8,
    #     polar = True,
    #     unif_x = False,
    #     v_radial = False,
    #     V0 = (0,0,0),
    #     dx = (200,200,200),
    #     dV = (80,80,80),
    #     center = True,
    # ),
    # cart_iso = dict(
    #     accs_train = (0,),
    #     accs_test = (0,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = True,
    #     n_turns = 0,
    #     t_mean = 30,
    #     t_sigma = 8,
    #     polar = False,
    #     unif_x = False,
    #     v_radial = False,
    #     V0 = (0,0,0),
    #     dx = (1000,1000,1000),
    #     dV = (80,80,80),
    # ),
    # cartesian = dict(
    #     accs_train = (0,),
    #     accs_test = (0,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = True,
    #     n_turns = 0,
    #     t_mean = 30,
    #     t_sigma = 8,
    #     polar = False,
    # ),
    # isotropic = dict(
    #     accs_train = (0,),
    #     accs_test = (0,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = True,
    #     n_turns = 0,
    #     t_mean = 30,
    #     t_sigma = 8,
    #     polar = True,
    #     unif_x = False,
    #     v_radial = False,
    #     V0 = (0,0,0),
    #     dx = (1000,1000,1000),
    #     dV = (80,80,80),
    # ),
    centered = dict(
        accs_train = (0,),
        accs_test = (0,),
        n_train = 1500,
        n_test = 50,
        n_targets_test = 20,
        old_mode = True,
        n_turns = 0,
        t_mean = 30,
        t_sigma = 8,
        polar = True,
        unif_x = False,
        dx = (200,200,200),
        center = True,
    ),
    const_speed = dict(
        accs_train = (0,),
        accs_test = (0,),
        n_train = 1500,
        n_test = 50,
        n_targets_test = 20,
        old_mode = True,
        n_turns = 0,
        t_mean = 30,
        t_sigma = 8,
        polar = True,
    ),
    acceleration = dict(
        accs_train = (16,),
        accs_test = (16,),
        n_train = 1500,
        n_test = 50,
        n_targets_test = 20,
        old_mode = True,
        n_turns = 0,
        t_mean = 30,
        t_sigma = 8,
        polar = True,
    ),
    # turns = dict(
    #     accs_train = (16,),
    #     accs_test = (16,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = False,
    #     n_turns = 1,
    #     t_mean = 30,
    #     t_sigma = 8,
    #     polar = True,
    # ),
    # turns_cent = dict(
    #     accs_train = (16,),
    #     accs_test = (16,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = False,
    #     n_turns = 1,
    #     t_mean = 30,
    #     t_sigma = 8,
    #     polar = True,
    #     unif_x = False,
    #     dx = (200,200,200),
    #     center = True,
    # ),
    # multiphase = dict(
    #     accs_train = (48,),
    #     accs_test = (48,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = False,
    #     n_turns = 4,
    #     t_mean = 15,
    #     t_sigma = 2.5,
    #     polar = True,
    # ),
    multiphase = dict(
        accs_train = (32,),
        accs_test = (32,),
        n_train = 1500,
        n_test = 50,
        n_targets_test = 20,
        old_mode = True,
        n_turns = 2,
        t_mean = 10,
        t_sigma = 2,
        polar = True,
    ),
    # multiphase_centered = dict(
    #     accs_train = (32,),
    #     accs_test = (32,),
    #     n_train = 1500,
    #     n_test = 50,
    #     n_targets_test = 20,
    #     old_mode = True,
    #     n_turns = 2,
    #     t_mean = 10,
    #     t_sigma = 2,
    #     polar = True,
    #     unif_x = False,
    #     dx = (200,200,200),
    #     center = True,
    # ),
    multiacc = dict(
        accs_train = (32,),
        accs_test = (16,32,64),
        n_train = 1500,
        n_test = 50,
        n_targets_test = 20,
        old_mode = True,
        n_turns = 2,
        t_mean = 10,
        t_sigma = 2,
        polar = True,
    ),
)


###########   SCENARIOS-GENERATING FUNCTIONS   ###########

def generate_scenarios(meta_args, load=False, **kwargs):
    args = {}
    for sc, margs in meta_args.items():
        print(f'\n{sc:}')
        args[sc] = get_scenario_args(**margs)
        E = gen_data(sc, args[sc], margs['n_train'], generate=not load, verbose=True, **kwargs)
    return args

def get_scenario_args(accs_train, accs_test, n_train, n_test, n_targets_test, **kwargs):
    sc = {}
    sc['train'] = get_group_settings(accs_train, n_train, seed_base=0, n_targets=1, train=True, **kwargs)
    sc['test'] = get_group_settings(accs_test, n_test, seed_base=n_train, n_targets=n_targets_test, train=False, **kwargs)
    return sc

def get_group_settings(max_acc=0, n_episodes=1000, n_targets=1, n_turns=3, p_vertical=0.3, rand_dir=True, interval_probs=None,
                       t_mean=None, t_sigma=None, polar=True, seed_base=0, old_mode=True, center=False, noise=1, far=0,
                       unif_x=True, v_radial=True, V0=(70,0,90), dx=(4e3,4e3,4e2), dV=None, title=None, train=False):
    if dV is None: dV = (15,rand_dir*10000,6)
    if title is None:
        title = 'train' if train else 'test'
    seeds = seed_base + np.arange(n_episodes)

    return dict(
        targets = dict(
            n_episodes = n_episodes,
            dt = 1,
            old_mode = old_mode,
            center = center,
            acc = max_acc,
            interval_probs = interval_probs,
            n_targets_args = dict(n1=n_targets-3, n2=n_targets+4) if n_targets>3 else n_targets,
            init_args=dict(t0=10, dt=3, X0=(0,0,0), dx=dx, V0=V0, dV=dV,
                           unif_x=unif_x, unif_v=False, v_radial=v_radial),
            n_turns_args = dict(n1=n_turns-1, n2=n_turns+2) if n_turns>1 else n_turns,
            line_args = dict(p_acc=0.5, t_mean=12 if t_mean is None else t_mean, t_sigma=3 if t_sigma is None else t_sigma,
                             t_mean_acc=6 if t_mean is None else t_mean, t_sigma_acc=2 if t_sigma is None else t_sigma, vmax0=150),
            turn_args = dict(p_vertical=p_vertical, p_left=0.5, a_mean=45, a_sigma=270, vertical_fac=0.1),
            seeds = seeds,
            title = title,
        ),
        radar = dict(
            noise_factor = noise,
            polar = polar,
            FAR = far,
        )
    )

def gen_data(title, groups_args, train_episodes, generate=False, verbose=True, tit_dict=None):
    if generate:
        E = PLAB.create_experiment(scenarios=groups_args, title=title, load=False)
        E.meta_episodes.loc[E.meta_episodes.seed>=(train_episodes), 'group'] = 1
        E.meta_targets.loc[E.meta_targets.seed_ep>=(train_episodes), 'group'] = 1
        E.save_data()
        X, Y, scenarios_train = PLAB.get_group_data(E, 0)
        X2, Y2, scenarios_test = PLAB.get_group_data(E, 1)
        PLAB.save_data(X, Y, scenarios_train, fname=f'{title}_train')
        PLAB.save_data(X2, Y2, scenarios_test, fname=f'{title}_test')
        if verbose:
            print(len(X), len(Y), X[0].shape, Y[0].shape, np.min(E.meta_targets['T']), np.max(E.meta_targets['T']), Counter(scenarios_train))

    else:
        E = PLAB.create_experiment(title=title, load=True)
        X, Y, scenarios_train = PLAB.get_group_data(E, 0)
        X2, Y2, scenarios_test = PLAB.get_group_data(E, 1)
        if verbose:
            print(len(X), len(Y), X[0].shape, Y[0].shape, np.min(E.meta_targets['T']), np.max(E.meta_targets['T']), Counter(scenarios_train))

    if verbose:
        if tit_dict:
            title = tit_dict[title]

        dd, axs = E.targets_consistency_test()
        axs[0].set_title(title, fontsize=14)
        axs[1].set_title(title, fontsize=14)

        axs = utils.Axes(1,1)
        utils.plot_quantiles(E.meta_targets['T'], plot_mean=True, ax=axs[0])
        axs[0].set_xlabel('quantile [%]')
        axs[0].set_ylabel('target life-time')

        axs = E.show_episodes(scenarios=['test'], episodes_per_scenario=1, max_targets=24,
                              detailed_lab=False, axs_args=dict(axsize=(6,5)))
        axs[0][0].set_title(title, fontsize=16)
        axs[0][0].get_legend().remove()

        story, info = E.get_target_story()
        print(info)
        print(story)

    return E


###########   SCENARIOS ANALYSIS   ###########

def get_meta_args_summary(margs):
    mm = pd.DataFrame(dict(
        varying_H = len(margs) * [True],
        anisotropic = [('unif_x' not in a or a['unif_x']) or ('v_radial' not in a or a['v_radial']) \
                       for a in margs.values()],
        polar = [a['polar'] for a in margs.values()],
        uncentered = [('center' not in a or not a['center']) for a in margs.values()],
        acceleration = [max(a['accs_train'])>0 for a in margs.values()],
        turns = [a['n_turns']>0 for a in margs.values()],
        multiphase = [a['n_turns']>1 for a in margs.values()],
    ), index=[sc for sc in margs])
    return mm

def show_meta_args_summary(margs, ax=None, cols=None):
    x = get_meta_args_summary(margs)
    if cols:
        x = x.loc[:, [c in cols for c in x.columns]]
    ax = sns.heatmap(x, cmap='RdYlGn', cbar=False, linewidths=0.3, ax=ax)
    ax.tick_params(axis='x', which='major', labelsize=13, labelbottom=False, bottom=False, top=False, labeltop=True, rotation=90)
    ax.tick_params(axis='y', which='major', labelsize=13, left=False, rotation=0)
    return ax

def summarize_scenarios(scenarios, axs=None):
    # load scenarios
    dd = pd.DataFrame()
    dd_tar = pd.DataFrame()
    for sc in scenarios:
        x,y,groups = PLAB.load_data(fname=f'{sc}_train')
        d_tar = pd.DataFrame(dict(
            scenario = len(y) * [sc],
            length = [len(xx) for xx in y],
        ))
        d = pd.DataFrame(dict(
            scenario = [sc for xx in y for xxx in xx[:,0]],
            t = [t for xx in y for t in range(len(xx))],
            x_x = [xxx for xx in y for xxx in xx[:,0]],
            x_y = [xxx for xx in y for xxx in xx[:,1]],
            x_z = [xxx for xx in y for xxx in xx[:,2]],
            x_vx = [xxx for xx in y for xxx in xx[:,3]],
            x_vy = [xxx for xx in y for xxx in xx[:,4]],
            x_vz = [xxx for xx in y for xxx in xx[:,5]],
            z_x = [xxx for xx in x for xxx in xx[:,0]],
            z_y = [xxx for xx in x for xxx in xx[:,1]],
            z_z = [xxx for xx in x for xxx in xx[:,2]],
            z_dop = [xxx for xx in x for xxx in xx[:,3]],
        ))
        dd_tar = pd.concat((dd_tar, d_tar))
        dd = pd.concat((dd, d))

    # calculate stats
    dd['r'] = np.sqrt(dd.x_x**2 + dd.x_y**2 + dd.x_z**2)
    dd['acceleration'] = np.sqrt(dd.x_vx.diff()**2 + dd.x_vy.diff()**2 + dd.x_vz.diff()**2)
    dd.loc[dd.t.diff()!=1, 'acceleration'] = np.nan
    dd['speed'] = np.sqrt(dd.x_vx**2 + dd.x_vy**2 + dd.x_vz**2)
    dd['phi'] = 180/np.pi*np.arccos(dd.x_vz/dd.speed)
    dd['noise'] = np.sqrt((dd.z_x-dd.x_x)**2 + (dd.z_y-dd.x_y)**2 + (dd.z_z-dd.x_z)**2)

    # init axes
    sns.set(font_scale=1.2)
    if axs is None:
        axs = utils.Axes(6,3,axsize=(5.5,3.8))
    a = 0

    # plot
    rot = 30
    sns.boxplot(data=dd_tar, x='scenario', y='length', showmeans=True, ax=axs[a])
    axs[a].set_xticklabels(axs[a].get_xticklabels(), rotation=rot)
    a += 1
    sns.boxplot(data=dd, x='scenario', y='r', showmeans=True, ax=axs[a])
    axs[a].set_xticklabels(axs[a].get_xticklabels(), rotation=rot)
    a += 1
    sns.boxplot(data=dd, x='scenario', y='speed', showmeans=True, ax=axs[a])
    axs[a].set_xticklabels(axs[a].get_xticklabels(), rotation=rot)
    a += 1
    sns.boxplot(data=dd, x='scenario', y='acceleration', showmeans=True, ax=axs[a])
    axs[a].set_xticklabels(axs[a].get_xticklabels(), rotation=rot)
    a += 1
    sns.boxplot(data=dd, x='scenario', y='phi', showmeans=True, ax=axs[a])
    axs[a].set_xticklabels(axs[a].get_xticklabels(), rotation=rot)
    a += 1
    utils.compare_quantiles(dd, 'scenario', 'noise', 'r', axs=[axs[a]], mean=None, box=True, showfliers=False)
    axs[a].set_xticklabels(axs[a].get_xticklabels(), rotation=rot)
    a += 1

    sns.set(font_scale=1)
    plt.tight_layout()
    return axs, dd, dd_tar
