'''
Written by Ido Greenberg, 2020
'''

from pathlib import Path
import pickle as pkl
from time import time
from warnings import warn
import multiprocessing as mp
import os, sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import seaborn as sns

import utils
import TargetSimulator as Sim
import SensorGenerator as Radar
import Trackers

DATA_PATH = Path('data/')


class Experiment:

    def __init__(self, solvers=None, seed=1, initialize=False, dt=1, title='tmp'):
        self.title = title
        self.dt = dt
        self.seed = seed
        self.set_seed(self.seed)

        self.scenarios = []         # list of scenarios names

        self.meta_episodes = None   # table of episodes (within scenarios)
        self.meta_targets = None    # table of targets (within episodes within scenarios)
        self.targets = {}           # targets[scenario][episode][target] = table of time, state & observations
        self.target_args = {}       # target_args[scenario][episode][target] = args of trajectory
        self.target_meta_args = {}  # target_meta_args[scenario][episode][target] = meta_args for random generation of args of trajectory
        self.n_episodes = {}        # n_episodes[scenario] = number of episodes

        self.radars = {}            # radar[scenario] = object that generates noisy detections from trajectory
        self.detections = {}        # detections[scenario][episode] = table of time, detection & original target
        self.radar_args = {}        # radar_args[scenario] = args of radar

        self.solvers = {}
        self.solvers_args = Experiment.get_default_solvers() if solvers is None else solvers
        self.default_solvers_order = []

        self.res = None

        if initialize:
            self.set_scenarios()
            self.initialize_solvers()

    #########   GENERAL METHODS   #########

    def set_seed(self, seed=None, use_global_as_default=False):
        if seed is None:
            if use_global_as_default:
                seed = self.seed
            else:
                return
        np.random.seed(seed)

    def save_data(self, filename=None, base_path=DATA_PATH):
        if filename is None: filename = self.title
        if not filename.endswith('.pkl'): filename += '.pkl'
        with open(base_path / filename, 'wb') as fd:
            pkl.dump((self.scenarios, self.target_meta_args, self.n_episodes,
                      self.meta_episodes, self.meta_targets, self.targets, self.target_args,
                      self.radars, self.radar_args, self.detections, self.res), fd)

    def load_data(self, filename=None, base_path=DATA_PATH, inplace=True):
        if filename is None: filename = self.title
        if not filename.endswith('.pkl'): filename += '.pkl'
        with open(base_path / filename, 'rb') as fd:
            scenarios, meta_args, n_eps, meta_eps, meta_targs, targets, tar_args, radars, rad_args, dets, res = pkl.load(fd)

        if inplace:
            self.scenarios, self.target_meta_args, self.n_episodes, self.meta_episodes, self.meta_targets, self.targets, self.target_args, self.radars, self.radar_args, self.detections, self.res = \
                scenarios, meta_args, n_eps, meta_eps, meta_targs, targets, tar_args, radars, rad_args, dets, res
        else:
            return scenarios, meta_args, n_eps, meta_eps, meta_targs, targets, tar_args, radars, rad_args, dets, res

    #########   SETUP METHODS   #########

    '''
    Simulation APIs:
    
    *** SEE DOCUMENTATION IN TargetSimulator.py FOR MEANING OF ARGUMENTS ***
    
    simulate_episodes(
        n_episodes=2, dt=1, acc=40, n_targets_args=2,
        init_args=None, n_turns_args=1, line_args=None, turn_args=None,
        seeds=None, title='scenario'
    )
    
    Simplified_Sensor_Generator(noise_factor=1, noise_base=(0.1,0.6,10,10))
    '''

    @staticmethod
    def default_targets_args(mode=0):
        # SEE DOCUMENTATION IN TargetSimulator.py FOR MEANING OF ARGUMENTS
        if mode == 0:
            return dict(
                n_episodes = 1000,
                dt = 1,
                acc = 40,
                n_targets_args = 1,
                init_args=dict(t0=10, dt=2, X0=(0,0,0), dx=200, V0=(100,45,80), dV=(0,0,0)),
                n_turns_args = 0,
                line_args = dict(p_acc=1, t_mean=10, t_sigma=1.5),
                turn_args = dict(p_left=0.5, a_mean=45, a_sigma=10),
                seeds = None,
                title = 'Shifted lines',
            )
        if mode == 1:
            return dict(
                n_episodes = 1000,
                dt = 1,
                acc = 40,
                n_targets_args = 1,
                init_args=dict(t0=10, dt=2, X0=(0,0,0), dx=200, V0=(100,45,80), dV=(20,100,8)),
                n_turns_args = 0,
                line_args = dict(p_acc=0.7, t_mean=10, t_sigma=2),
                turn_args = dict(p_left=0.5, a_mean=45, a_sigma=10),
                seeds = None,
                title = 'Straight lines',
            )
        raise ValueError(mode)

    @staticmethod
    def default_radar_args(noise_factor=1, FAR=0):
        return dict(
            noise_base = (0.1, 0.6, 10, 10),
            noise_factor = noise_factor,
            FAR = FAR,
        )

    def set_scenarios(self, scenarios=None, targets_mode=0, radar_noise=0,
                      enforce_consistent_names=True):
        # scenarios['targets'] = targets_args, scenarios['radar'] = radar_args
        if scenarios is None:
            targets = Experiment.default_targets_args(targets_mode)
            nm = targets['title']
            scenarios = {nm: dict(
                targets = targets,
                radar = Experiment.default_radar_args(radar_noise),
            )}

        if enforce_consistent_names:
            for k,v in scenarios.items():
                v['targets']['title'] = k

        self.scenarios = list(scenarios.keys())
        for s in self.scenarios:
            self.target_meta_args[s] = scenarios[s]['targets'] if 'targets' in scenarios[s] else \
                Experiment.default_targets_args(targets_mode)
            self.radar_args[s] = scenarios[s]['radar'] if 'radar' in scenarios[s] else \
                Experiment.default_radar_args(radar_noise)

    @staticmethod
    def get_default_solvers(matcher_args=None):
        if matcher_args is None: matcher_args = {}
        matcher = Experiment.get_default_matcher(**matcher_args)
        trackers = Experiment.get_default_trackers()
        return {nm:(matcher, tracker) for nm,tracker in trackers.items()}

    @staticmethod
    def get_default_trackers():
        return dict(
            naive = (Trackers.NaiveTracker, dict()),
            KF = (Trackers.KF_default, dict()),
        )

    @staticmethod
    def get_default_matcher(match_thresh=0):
        return (Trackers.HungarianAssigner, dict(match_thresh=match_thresh))

    def initialize_solvers(self):
        self.solvers = {}
        for nm, (matcher, tracker) in self.solvers_args.items():
            self.solvers[nm] = Trackers.Solver(matcher=matcher[0], matcher_args=matcher[1],
                                               tracker=tracker[0], tracker_args=tracker[1], dt=self.dt)
        self.default_solvers_order = list(self.solvers.keys())

    #########   SCENARIOS SIMULATION   #########

    def generate_scenarios(self, to_save=None, verbose=1):
        self.meta_episodes = pd.DataFrame()
        self.meta_targets = pd.DataFrame()
        for s in self.scenarios:
            t0 = time()
            if verbose >= 1:
                print(f'Generating scenario {s:s}... ', end='')

            dd, targets, target_args = Sim.simulate_episodes(**self.target_meta_args[s])
            mm = Sim.meta_targets2episodes(dd)

            self.meta_episodes = pd.concat((self.meta_episodes, mm))
            self.meta_targets = pd.concat((self.meta_targets, dd))
            self.targets[s] = targets
            self.target_args[s] = target_args
            self.n_episodes[s] = len(targets)

            self.radars[s] = Radar.Radar(**self.radar_args[s])
            self.detections[s] = [self.radars[s].make_episode_detections(episode_targets, update_targets=True) \
                                  for episode_targets in self.targets[s]]

            if verbose >= 1:
                print(f'done.\t({time()-t0:.0f} [s])')

        self.meta_episodes.reset_index(drop=True, inplace=True)
        self.meta_targets.reset_index(drop=True, inplace=True)

        if to_save:
            self.save_data(to_save)

    #########   TRACKING SIMULATION   #########

    def clean_res(self, scenario, episodes, solvers):
        for episode in episodes:
            targets = self.targets[scenario][episode]
            detections = self.detections[scenario][episode]
            for solver_nm in solvers:
                for target in targets:
                    for dm in ('x','y','z'):
                        target[f'{solver_nm:s}_{dm:s}'] = np.nan
                    target[f'match_{solver_nm:s}'] = -1
                    target[f'SE_{solver_nm:s}'] = np.nan
                detections[f'match_{solver_nm:s}'] = -1

    def run_scenarios(self, scenarios=None, solvers=None, do_save=False, group=None, subsample=1,
                      base_path=DATA_PATH, print_freq=5, distributed=True, gb_per_job=3, verbose=3):
        if scenarios is None: scenarios = self.scenarios
        if solvers is None: solvers = list(self.solvers.keys())
        filt = None if group is None else GroupFilter(self.meta_episodes, group)
        pd.options.mode.chained_assignment = None  # default='warn'
        if distributed:
            P = mp.Pool()

        # initialize results columns
        ids = self.meta_episodes.scenario.isin(scenarios)
        if filt is not None:
            ids = ids & (self.meta_episodes.group == group)
        if subsample < 1:
            ids = ids & (np.random.random(len(ids))<subsample)
        for goal in ('RMSE','match_rate'):
            for sol in solvers:
                col = f'{goal:s}_{sol:s}'
                if col in self.meta_episodes:
                    self.meta_episodes.loc[ids, f'{goal:s}_{sol:s}'] = sum(ids) * [np.nan]
                else:
                    self.meta_episodes[f'{goal:s}_{sol:s}'] = len(self.meta_episodes) * [np.nan]

        # run scenarios
        t0 = time()
        for sc in scenarios:
            if verbose >= 1:
                print(f'Running {sc:s}...', end='')
                if verbose >= 2:
                    print()

            episodes = [ep for ep in range(self.n_episodes[sc]) if filt is None or filt.isin(sc,ep)]
            N = len(episodes)
            self.clean_res(sc, episodes, solvers)

            n = Experiment.choose_procs_num(gb_per_job=gb_per_job, verbose=verbose-1)
            if distributed and n>1 and N>0:
                dist_episodes = [episodes[i*N//n : (i+1)*N//n] for i in range(n)]
                args = [[self, solvers, sc, i, n, dist_episodes[i], print_freq, verbose] for i in range(n)]
                results = P.map(Experiment.run_episodes_thread, args)
                self.store_results(results)

            else:
                for i,ep in enumerate(episodes):
                    if verbose >= 2 and (i%print_freq)==0:
                        print(f'\tEpisode {i:03d}/{len(episodes):03d}...', end='')
                    for sol in solvers:
                        self.run_episode(sc, ep, sol)
                    if verbose >= 2 and (i%print_freq)==0:
                        print(f' done.\t({time() - t0:.0f} [s])')
            if verbose >= 1:
                print(f' done.\t({time() - t0:.0f} [s])')

        if distributed:
            P.close()

        # summarize results
        self.update_results_summary(scenarios, solvers, filt=filt)
        pd.options.mode.chained_assignment = 'warn'

        if do_save:
            if isinstance(do_save, str):
                nm = do_save
            else:
                nm = self.title + '_res'
            self.save_data(nm, base_path)

    @staticmethod
    def choose_procs_num(nmax=None, nmin=1, gb_per_job=4, buffer_in_gb=8, verbose=1):
        if nmax is None:
            nmax = os.cpu_count()
        if 'linux' in sys.platform:
            with open('/proc/meminfo') as file:
                for line in file:
                    if 'MemFree' in line:
                        free_mem_in_kb = float(line.split()[1])
                        break
            free_mem_in_gb = free_mem_in_kb / (1024**2)
            n_free = int(max(nmin, (free_mem_in_gb-buffer_in_gb) / gb_per_job) )
            if n_free < nmax:
                if verbose >= 1:
                    print(f'Reduced distribution to {n_free:d} cores to fit into {free_mem_in_gb:.0f}GB.')
            n = min(nmax, n_free)
        else:
            n = nmax
        return n

    @staticmethod
    def run_episodes_thread(args):
        self, solvers, sc, i, n, episodes, print_freq, verbose = args

        if verbose >= 2:
            print(f'\tBatch {i:d} starts...')
        t0 = time()
        res = dict(sc=[], ep=[], sol=[], row=[], t=[], tar=[], trk=[], x=[], y=[], z=[], se=[])
        solver_records = {sol:{} for sol in solvers}
        for j,ep in enumerate(episodes):
            for sol in solvers:
                self.run_episode(sc, ep, sol, res=res)
            if verbose >= 3 and ((j+1)%print_freq)==0 and j!=len(episodes)-1:
                print(f'\t\t[{i:02d}/{n:02d}] episode {j+1:03d}/{len(episodes):03d} done.\t({time()-t0:.1f} [s])')
        if verbose >= 2:
            print(f'\tBatch {i:d} done.\t({time()-t0:.1f} [s])')

        for sol in solvers:
            solver_records[sol]['targets_record'] = self.solvers[sol].targets_record
            solver_records[sol]['runtime_record'] = self.solvers[sol].runtime_record

        return res, solver_records

    def run_episode(self, scenario, episode, solver_nm, res=None):
        pd.options.mode.chained_assignment = None  # default='warn'
        # initialization
        solver = self.solvers[solver_nm]
        detections = self.detections[scenario][episode]
        detections_stream = detections.groupby('t').apply(lambda d: (
            [d.iloc[i,:5] for i in range(len(d))],
            [d.target.values[i] for i in range(len(d))]
        ))
        times = detections_stream.keys()
        times = np.arange(np.min(times), np.max(times)+self.dt/2, self.dt)

        # run episode
        solver.initialize()
        for i, t in enumerate(times):
            dets, tars = detections_stream[t] if t in detections_stream else ([], [])
            assigned_targets = solver.step(dets, tars)
            self.update_assignments(scenario, episode, solver_nm, t, tars, assigned_targets, res=res)

        return res

    def update_assignments(self, scenario, episode, solver_nm, t,
                           curr_targets, assigned_targets, res=None):
        if res==False:
            return
        col = f'match_{solver_nm:s}'
        targets = self.targets[scenario][episode]
        detections = self.detections[scenario][episode]
        assignments = {tar:trk for tar,trk in zip(curr_targets,assigned_targets)}

        for tar, trk in assignments.items():
            if tar < 0:
                continue

            tracker = self.solvers[solver_nm].get_tracker_by_id(trk)
            if tracker is None:
                warn(f"Cannot find target's tracker ({solver_nm}, {scenario}/{episode:d}/{tar:d}).")
                # import pdb
                # pdb.set_trace()
                continue
            xyz = tracker.get_pos()
            target = targets[tar]
            SE = np.sum((np.array((target.x_x[t],target.x_y[t],target.x_z[t])) - xyz) ** 2)
            row = np.where((detections.t==t) & (detections.target==tar))[0][0]

            if res is None:
                # update inplace
                # update in targets
                if col not in target.columns:
                    # TODO move to more efficient location
                    target[col] = np.nan
                    for j,dm in enumerate(('x','y','z')):
                        target[f'{solver_nm:s}_{dm:s}'] = np.nan
                    target[f'SE_{solver_nm:s}'] = np.nan
                target[col][t] = trk
                for j,dm in enumerate(('x','y','z')):
                    target[f'{solver_nm:s}_{dm:s}'][t] = xyz[j]
                target[f'SE_{solver_nm:s}'][t] = SE
                # update in detections
                if col not in detections.columns:
                    detections[col] = np.nan
                detections[col][row] = trk

            else:
                row = np.where((detections.t==t) & (detections.target==tar))[0][0]
                res['sc'].append(scenario)
                res['ep'].append(episode)
                res['sol'].append(solver_nm)
                res['row'].append(row)
                res['t'].append(t)
                res['tar'].append(tar)
                res['trk'].append(trk)
                res['x'].append(xyz[0])
                res['y'].append(xyz[1])
                res['z'].append(xyz[2])
                res['se'].append(SE)

        return res

    def store_results(self, results):
        # store run outputs in targets & detections
        for res, records in results:
            for sc,ep,sol,row,t,tar,trk,x,y,z,se in zip(
                    *[res[k] for k in ('sc','ep','sol','row','t','tar','trk','x','y','z','se')]):
                target = self.targets[sc][ep][tar]
                target[f'match_{sol:s}'][t] = trk
                target[f'{sol:s}_x'][t] = x
                target[f'{sol:s}_y'][t] = y
                target[f'{sol:s}_z'][t] = z
                target[f'SE_{sol:s}'][t] = se
                self.detections[sc][ep][f'match_{sol:s}'][row] = trk

            for sol in records:
                for k in self.solvers[sol].targets_record:
                    self.solvers[sol].targets_record[k].extend(records[sol]['targets_record'][k])
                for k in self.solvers[sol].runtime_record:
                    self.solvers[sol].runtime_record[k].extend(records[sol]['runtime_record'][k])

    def update_results_summary(self, scenarios=None, solvers=None, filt=None):
        if scenarios is None: scenarios = self.scenarios
        if solvers is None: solvers = list(self.solvers.keys())

        # update targets data-frame
        ids = []
        match_rates = {sol:[] for sol in solvers}
        RMSEs = {sol:[] for sol in solvers}
        for sc in scenarios:
            for ep, targets in enumerate(self.targets[sc]):
                if filt is not None and not filt.isin(sc,ep):
                    continue
                row = np.where((self.meta_episodes.scenario==sc) & (self.meta_episodes.episode==ep))[0][0]
                ids.append(row)
                for solver_nm in solvers:
                    match_col = f'match_rate_{solver_nm:s}'
                    err_col = f'RMSE_{solver_nm:s}'
                    match_rate = 0
                    RMSE = 0
                    count = 0
                    # na_count = 0
                    for i, target in enumerate(targets):
                        T = len(target)
                        # matches = 100*(target[f'match_{solver_nm:s}']==i).mean()
                        # matches = 100 * (np.diff(target[f'match_{solver_nm:s}']) == 0).mean()
                        matches = 100 * np.bincount(target[f'match_{solver_nm:s}']).max() / T
                        match_rate += matches
                        if matches == 0:
                            warn(f'No matches at all: {solver_nm}, {sc}/{ep}/{i:d}')
                            # import pdb
                            # pdb.set_trace()
                        sse = target[f'SE_{solver_nm:s}'].sum()
                        if not np.isnan(sse):
                            RMSE += sse
                            count += T
                            # na_count += len(target)
                        if match_col not in self.meta_targets:
                            self.meta_targets[match_col] = 0
                        if err_col not in self.meta_targets:
                            self.meta_targets[err_col] = np.inf
                        row = np.where((self.meta_targets.scenario==sc) & (self.meta_targets.episode==ep) & (self.meta_targets.target==i))[0][0]
                        self.meta_targets[match_col].values[row] = matches
                        self.meta_targets[err_col].values[row] = np.sqrt(sse/T)

                    match_rate /= len(targets)
                    RMSE = np.sqrt(RMSE/count)
                    RMSEs[solver_nm].append(RMSE)
                    match_rates[solver_nm].append(match_rate)

        # update episodes data-frame
        for sol in solvers:
            self.meta_episodes[f'RMSE_{sol:s}'].values[ids] = RMSEs[sol]
            self.meta_episodes[f'match_rate_{sol:s}'].values[ids] = match_rates[sol]

        self.res = self.get_all_errors(scenarios, solvers, filt=filt)


    #########   ANALYSIS METHODS   #########

    def show_episodes(self, scenarios=None, episodes_per_scenario=2, **kwargs):
        if scenarios is None: scenarios = self.scenarios
        axs = []
        for sc in scenarios:
            axs.append(Sim.show_episodes(self.targets[sc], self.target_args[sc], max_episodes=episodes_per_scenario, **kwargs))
        return axs

    def show_detections(self, scenarios=None):
        if scenarios is None: scenarios = self.scenarios
        for sc in scenarios:
            Radar.detections_summary(self.detections[sc][0], self.targets[sc][0], tit=sc)

    def get_score_per_solver(self, unit='episode', goal='RMSE', scenarios=None, group=None):
        dct = dict(
            episode = self.meta_episodes,
            target = self.meta_targets
        )
        dd = dct[unit]
        if scenarios is not None:
            dd = dd[dd.scenario.isin(scenarios)]
        if group is not None:
            dd = dd[dd.group==group]
        cols = [c for c in dd.columns if c.startswith(goal)]
        return utils.pd_merge_cols(dd, cols_to_merge=cols, values_cols=goal, case_col='solver',
                                   cases_names=[c[len(goal)+1:] for c in cols])

    def analyze_results(self, scenarios=None, solvers=None, ref_solver=None, axs=None, axs_args=None, group=None, target_classes=None,
                        plot_trajs=None, plot_trajs_3d=None, show_outliers=False, fast=True, large=None):
        # input pre-processing
        if scenarios is None: scenarios = self.scenarios
        print('Scenarios:', scenarios)
        if solvers is None:
            solvers = [c[len('RMSE_'):] for c in self.meta_episodes.columns if c.startswith('RMSE')]
            solvers = [sol for sol in self.default_solvers_order if sol in solvers]
        if ref_solver is None: ref_solver = solvers[0]
        if target_classes is None: target_classes = sorted(np.unique(self.meta_targets.target_class))
        if plot_trajs is None: plot_trajs = (4,1) if len(scenarios)<=2 else (2,1)
        if plot_trajs_3d is None: plot_trajs_3d = (2,1) if len(scenarios)<=2 else (1,1)
        if axs is None:
            if axs_args is None: axs_args = {}
            if large is None:
                large = [0,1,2][max((len(scenarios)>4)+(len(scenarios)>8),len(solvers)>3)+(len(solvers)>6)]
            W = [4,2,1][large]
            axs_args = utils.update_dict(axs_args, dict(W=W, axsize=(16/W,max(3.5,0.6*16/W)), fontsize=16))
            axs = utils.Axes(4 + 4 + 3*(0*(len(solvers)-1)+4) + 2 + 8 + (2*len(scenarios)*np.prod(plot_trajs) if plot_trajs else 0) + plot_trajs_3d[0],
                             **axs_args)
        a = 0

        # RMSE per episode
        res_ep = self.get_score_per_solver('episode', scenarios=scenarios, group=group)
        sns.boxplot(data=res_ep, x='scenario', hue='solver', y='RMSE', hue_order=solvers,
                    showmeans=True, showfliers=show_outliers, ax=axs[a])
        utils.labels(axs[a], 'Scenario', 'RMSE per episode', fontsize=16)
        axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=13, rotation=20)
        utils.overwrite_legend(
            axs[a], lambda sol:f'{sol:s} (RMSE={np.sqrt((res_ep[res_ep.solver==sol].RMSE**2).mean()):.0f})',
            lambda s: s in set(res_ep.solver), 12)
        a += 1

        for sol in solvers:
            lab = f'{sol:s} (RMSE={np.sqrt((res_ep[res_ep.solver==sol].RMSE**2).mean()):.0f})'
            utils.plot_quantiles(res_ep[res_ep.solver == sol].RMSE, axs[a], plot_mean=True, label=lab)
        utils.labels(axs[a], 'Episode quantile [%]', 'RMSE', fontsize=16)
        axs[a].legend(fontsize=12)
        axs[a].set_yscale('log')
        a += 1

        # RMSE per target
        res_tar = self.get_score_per_solver('target', scenarios=scenarios, group=group)
        sns.boxplot(data=res_tar, x='target_class', hue='solver', y='RMSE', hue_order=solvers, order=target_classes,
                    showmeans=True, showfliers=show_outliers, ax=axs[a])
        utils.labels(axs[a], 'Scenario', 'RMSE per target', fontsize=16)
        axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=13, rotation=20)
        utils.overwrite_legend(
            axs[a], lambda sol:f'{sol:s} (RMSE={np.sqrt((res_tar[res_tar.solver==sol].RMSE**2).mean()):.0f})',
            lambda s: s in set(res_tar.solver), 12)
        a += 1

        for sol in solvers:
            lab = f'{sol:s} (RMSE={np.sqrt((res_tar[res_tar.solver==sol].RMSE**2).mean()):.0f})'
            utils.plot_quantiles(res_tar[(res_tar.solver==sol) & (res_tar.RMSE.notna())].RMSE, axs[a],
                                 plot_mean=True, label=lab)
        utils.labels(axs[a], 'Target quantile [%]', 'RMSE', fontsize=16)
        axs[a].legend(fontsize=12)
        axs[a].set_yscale('log')
        a += 1

        # Match-rate per episode
        res_ep = self.get_score_per_solver('episode', 'match_rate', scenarios=scenarios, group=group)
        sns.boxplot(data=res_ep, x='scenario', hue='solver', y='match_rate', hue_order=solvers,
                    showmeans=True, showfliers=show_outliers, ax=axs[a])
        utils.labels(axs[a], 'Scenario', 'Correct matches per episode [%]', fontsize=14)
        axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=13, rotation=20)
        utils.overwrite_legend(
            axs[a], lambda sol:f'{sol:s} (RMSE={res_ep[res_ep.solver==sol].match_rate.mean():.1f}%)',
            lambda s: s in set(res_ep.solver), 12)
        a += 1

        for sol in solvers:
            lab = f'{sol:s} ({res_ep[res_ep.solver==sol].match_rate.mean():.1f}%)'
            utils.plot_quantiles(res_ep[res_ep.solver==sol].match_rate, axs[a], plot_mean=True, label=lab)
        utils.labels(axs[a], 'Episode quantile [%]', 'Correct matches [%]', fontsize=16)
        axs[a].legend(fontsize=12)
        a += 1

        # Match-rate per target
        res_tar = self.get_score_per_solver('target', 'match_rate', scenarios=scenarios, group=group)
        sns.boxplot(data=res_tar, x='target_class', hue='solver', y='match_rate', hue_order=solvers, order=target_classes,
                    showmeans=True, showfliers=show_outliers, ax=axs[a])
        utils.labels(axs[a], 'Scenario', 'Correct matches per target [%]', fontsize=14)
        axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=13, rotation=20)
        utils.overwrite_legend(
            axs[a], lambda sol:f'{sol:s} ({res_tar[res_tar.solver==sol].match_rate.mean():.1f}%)',
            lambda s: s in set(res_tar.solver), 12)
        a += 1

        for sol in solvers:
            lab = f'{sol:s} ({res_tar[res_tar.solver==sol].match_rate.mean():.1f}%)'
            utils.plot_quantiles(res_tar[res_tar.solver==sol].match_rate, axs[a], plot_mean=True, label=lab)
        utils.labels(axs[a], 'Target quantile [%]', 'Correct matches [%]', fontsize=16)
        axs[a].legend(fontsize=12)
        a += 1

        # Solvers pairs comparison
        n = 0*(len(solvers)-1)+4
        for goal, unit in (('RMSE','m'), ('MSE','m^2'), ('match_rate','%')):
            self.compare_solvers(ref_solver, solvers=solvers, group=group, goal=goal, goal_units=unit,
                                 target_classes=target_classes, axs=[axs[a+i] for i in range(n)])
            a += n

        # Runtime analysis
        try:
            self.runtime_analysis(solvers, [axs[a],axs[a+1]], show_outliers=show_outliers)
        except KeyError:
            warn('Could not find data for running-time analysis.')
        a += 2

        # In-target stats
        self.errors_analysis(scenarios=scenarios, solvers=solvers, group=group, fast=fast, axs=[axs[a+i] for i in range(9)])
        a += 8

        if plot_trajs:
            n = len(scenarios) * int(np.prod(plot_trajs))
            self.plot_trajectories(scenarios=scenarios, solvers=solvers, n_episodes=plot_trajs[0], n_targets=plot_trajs[1],
                                   group=group, axs=[axs[a+i] for i in range(n)])
            a += n
            self.plot_trajectories(scenarios=scenarios, solvers=solvers, n_episodes=plot_trajs[0], n_targets=plot_trajs[1],
                                   dims=('x','z'), group=group, axs=[axs[a+i] for i in range(n)])
            a += n

        plt.tight_layout()

        for ep in range(plot_trajs_3d[0]):
            self.plot_vertical_trajectory(scenarios[0], ep, solvers=solvers, ax=axs[a])
            a += 1

        if plot_trajs_3d:
            self.plot_trajectories_3d(scenarios=scenarios, solvers=solvers, n_episodes=plot_trajs_3d[0], n_targets=plot_trajs_3d[1],
                                      group=group)

        # Advanced results-slicing
        self.slice_results(solvers, scenarios, goal='error')

        # Advanced sample analysis
        self.target_analysis(scenarios[0],0,0, solvers=solvers, ref_solver=ref_solver)

        return axs

    def errors_analysis(self, scenarios=None, solvers=None, dims=('x','y','z'), group=None,
                        axs=None, axs_args=None, recalculate=False, fast=False):
        filt = None if group is None else GroupFilter(self.meta_episodes, group)
        if recalculate or self.res is None:
            self.res = self.get_all_errors(scenarios, solvers, dims, filt=filt)
        d = self.res
        if scenarios: d = d[d.scenario.isin(scenarios)]
        if solvers is None:
            solvers = [sol for sol in self.default_solvers_order if sol in d.solver.values]
        else:
            d = d[d.solver.isin(solvers)]
        # solvers = np.unique(d.solver)
        # scenario | solver | time_step | SE | bias_x | bias_y | bias_z
        dd = utils.pd_merge_cols(d, [c for c in d.columns if c.startswith('bias_')],
                                 'bias', 'dimension', dims)
        dd['error'] = np.abs(dd.bias)

        if axs is None:
            if axs is None:
                if axs_args is None:
                    axs_args = {}
                axs_args = utils.update_dict(axs_args, dict(W=3, axsize=(16/3, 3.5)))
            axs = utils.Axes(8, **axs_args)
        a = 0

        # error vs. time
        sns.lineplot(data=d, x='time_step', hue='solver', y='error', hue_order=solvers, ax=axs[a],
                     ci=None if fast else 95)
        utils.labels(axs[a], 'time-step', '|error|', None, 16)
        utils.overwrite_legend(
            axs[a], lambda sol:f'{sol:s} (RMSE={np.sqrt(d[d.solver==sol].SE.mean()):.0f})',
            lambda s: s in set(d.solver), 12)
        axs[a].set_yscale('log')
        a += 1

        sns.lineplot(data=d, x='target_progress', hue='solver', y='error', hue_order=solvers, ax=axs[a],
                     ci=None if fast else 95)
        utils.labels(axs[a], 'trajectory progress [%]', '|error|', None, 16)
        utils.overwrite_legend(
            axs[a], lambda sol:f'{sol:s} (RMSE={np.sqrt(d[d.solver==sol].SE.mean()):.0f})',
            lambda s: s in set(d.solver), 12)
        axs[a].set_yscale('log')
        a += 1

        # sns.boxplot(data=d, x='solver', hue='time_step', y='error', order=solvers,
        #             showmeans=True, showfliers=False, fliersize=0.4, ax=axs[a])
        # axs[a].get_legend().remove()
        # axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=14)
        # utils.labels(axs[a], 'solver / time-step', '|error| (outliers ignored)', None, 16)
        # a += 1

        # error vs. dimension
        sns.boxplot(data=dd, hue='solver', x='dimension', y='bias', hue_order=solvers,
                    showmeans=True, showfliers=False, fliersize=0.4, ax=axs[a])
        utils.labels(axs[a], 'solver', 'bias (outliers ignored)', None, 16)
        axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=14)
        a += 1

        sns.boxplot(data=dd, hue='solver', x='dimension', y='error', hue_order=solvers,
                    showmeans=True, showfliers=False, fliersize=0.4, ax=axs[a])
        utils.labels(axs[a], 'solver', '|error| (outliers ignored)', None, 16)
        axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=14)
        a += 1

        for dm in dims:
            utils.compare_quantiles(d, 'time_step', f'error_{dm:s}', 'solver', xbins=8, hbins=10, axs=[axs[a]])
            utils.labels(axs[a], 'time-step', f'{dm:s}-error', None, 16)
            a += 1

        # general
        sns.boxplot(data=d, x='scenario', hue='solver', y='error', hue_order=solvers,
                    showmeans=True, showfliers=False, fliersize=0.4, ax=axs[a])
        utils.labels(axs[a], 'Scenario', '|error| per detection\n(outliers ignored)', fontsize=16)
        axs[a].set_xticklabels(axs[a].get_xticklabels(), fontsize=13, rotation=20)
        utils.overwrite_legend(
            axs[a], lambda sol:f'{sol:s} (RMSE={np.sqrt(d[d.solver==sol].SE.mean()):.0f})',
            lambda s: s in set(d.solver), 12)
        a += 1

        return axs, d, dd

    def get_all_errors(self, scenarios=None, solvers=None, dims=('x','y','z'),
                       remove_nan_rows_for_all_solvers=True, filt=None):
        if scenarios is None: scenarios = self.scenarios
        if solvers is None: solvers = self.default_solvers_order
        self.meta_targets['identifier'] = [f'{sc}_{ep}_{tar}' for sc,ep,tar in zip(
            self.meta_targets.scenario, self.meta_targets.episode, self.meta_targets.target
        )]
        self.meta_targets.set_index(self.meta_targets.identifier, inplace=True)
        phase_class_fun = lambda intr: ('const' if intr[1]['acc']==0 else 'acc') if intr[0].__name__=='line' else \
            (['left','right'][intr[1]['angle']>=0] if not intr[1]['vertical'] else ['down','up'][intr[1]['angle']>=0])

        # get all targets
        targets = []
        for sc in scenarios:
            for ep in range(len(self.targets[sc])):
                if filt is not None and not filt.isin(sc,ep):
                    continue
                for tar in range(len(self.targets[sc][ep])):
                    target = self.targets[sc][ep][tar].copy()
                    target['scenario'] = sc
                    target['episode'] = ep
                    target['target'] = tar
                    target['identifier'] = f'{sc}_{ep}_{tar}'
                    target['tar_class'] = self.meta_targets.loc[f'{sc}_{ep}_{tar}','target_class']
                    phase_classes = [phase_class_fun(intr) for intr in self.target_args[sc][ep][tar]['intervals']]
                    target['phase_class'] = [phase_classes[p] for p in target.phase]
                    target['traj_len'] = len(target)
                    target['time_step'] = np.arange(len(target))
                    target['target_progress'] = np.round(100*(target.t-target.t.values[0])/(len(target)-1))
                    target['time_in_phase'] = 0
                    for t in range(1,len(target)):
                        target['time_in_phase'].values[t] = target.time_in_phase.values[t-1]+1 if target.phase.values[t]==target.phase.values[t-1] else 0
                    if remove_nan_rows_for_all_solvers:
                        ids = target.notna().all(axis=1)
                        target = target[ids]
                    targets.append(target)
        dd = pd.concat(targets)

        # merge all solvers to a single column
        cols = [[c for c in dd.columns if c.endswith('_' + dm) and not c in ('x_' + dm, 'z_' + dm)]
                for dm in dims]
        cols.append([c for c in dd.columns if c.startswith('SE_')])
        cols.append([c for c in dd.columns if c.startswith('match_')])
        dd = utils.pd_merge_cols(dd, cols, dims+('SE', 'match'), 'solver', [c[3:] for c in cols[-2]])

        # calculate biases & errors
        dd['R'] = np.sqrt(dd.x_x**2+dd.x_y**2+dd.x_z**2)
        dd['error'] = np.sqrt(dd.SE)
        for dm in dims:
            dd[f'bias_{dm:s}'] = dd[f'{dm:s}'] - dd[f'x_{dm:s}']
            dd[f'error_{dm:s}'] = np.abs(dd[f'bias_{dm:s}'])

        return dd

    def slice_results(self, solvers=None, scenarios=None, goal='error'):
        # recommended values for "goal": error, SE.
        if solvers is None: solvers = self.default_solvers_order
        if scenarios is None: scenarios = ['test']
        res = self.res[self.res.solver.isin(solvers)&self.res.scenario.isin(scenarios)]
        res0 = res[res.solver==solvers[0]]

        axs = utils.Axes(2,2, axsize=(6,4))
        target_classes = sorted(np.unique(self.meta_targets.target_class))
        sns.boxplot(data=res0, x='tar_class', y='time_step', order=target_classes, showmeans=True, ax=axs[0])
        sns.boxplot(data=res0, x='tar_class', y='R', order=target_classes, showmeans=True, ax=axs[1])
        plt.tight_layout()
        utils.compare_quantiles(res0, 'time_step', 'R', 'tar_class')

        utils.compare_quantiles(res, 'R', goal, 'solver', xbins=4)

        utils.compare_quantiles(res, 'time_step', goal, 'solver', 'tar_class')
        utils.compare_quantiles(res, 'R', goal, 'solver', 'tar_class')
        # utils.compare_quantiles(res, 'tar_class', goal, 'solver', 'R')
        utils.compare_quantiles(res, 'time_in_phase', goal, 'solver', 'phase_class')
        # utils.compare_quantiles(res, 'time_in_phase', goal, 'solver', 'tar_class')

    def plot_trajectories(self, scenarios=None, episodes=None, targets=None, n_episodes=2, n_targets=4,
                          dims=('x','y'), solvers=None, group=None, tmin=None, tmax=None, axs=None, axs_args=None):
        if scenarios is None: scenarios = self.scenarios
        if solvers is None: solvers = self.default_solvers_order
        if ('Naive' in self.default_solvers_order) and ('Naive' not in solvers): solvers = ['Naive'] + solvers # (for detections display)
        if episodes is None:
            episodes = {sc: np.arange(min(n_episodes, len(self.targets[sc]))) for sc in scenarios}
        if not isinstance(episodes, dict):
            episodes = {sc:episodes for sc in scenarios}
        if targets is None:
            targets = {sc:{ep:np.arange(min(n_targets, len(self.targets[sc][ep]))) for ep in episodes[sc]} for sc in scenarios}
        if not isinstance(targets, dict):
            targets = {sc:{ep:targets for ep in episodes[sc]} for sc in scenarios}
        filt = None if group is None else GroupFilter(self.meta_episodes, group)
        if axs is None:
            if axs_args is None:
                axs_args = {}
            axs_args = utils.update_dict(axs_args, dict(W=n_targets, axsize=(16/n_targets,3.5*4/n_targets)))
            N = sum([len(targets[sc][ep]) for sc in scenarios for ep in episodes[sc]]) if len(solvers)>1 else \
                sum([len(episodes[sc]) for sc in scenarios])
            axs = utils.Axes(N, **axs_args)

        a = 0
        for sc in scenarios:
            # generator = (targets for ep,targets in enumerate(self.targets[sc]) if filt is None or filt.isin(sc,ep))
            for ep in episodes[sc]: #i, episode in enumerate((generator)):
                if filt is not None and not filt.isin(sc,ep):
                    continue
                for tar in targets[sc][ep]:
                    cls = self.meta_targets[(self.meta_targets.scenario==sc)&(self.meta_targets.episode==ep)&(self.meta_targets.target==tar)].target_class.values[0]
                    target = self.targets[sc][ep][tar]
                    # plot target
                    axs[a].plot(target[f'x_{dims[0]}'].values[tmin:tmax], target[f'x_{dims[1]}'].values[tmin:tmax], 'k.-', linewidth=1, markersize=4)
                    axs[a].plot(target[f'x_{dims[0]}'].values[tmin:tmax][0], target[f'x_{dims[1]}'].values[tmin:tmax][0], 'k>', markersize=8)
                    axs[a].plot(target[f'x_{dims[0]}'].values[tmin:tmax][-1], target[f'x_{dims[1]}'].values[tmin:tmax][-1], 'ks', markersize=8)
                    # plot estimations
                    if len(solvers) > 1:
                        for sol in solvers:
                            rmse = np.sqrt(target[f'SE_{sol:s}'].mean())
                            axs[a].plot(target[f'{sol:s}_{dims[0]}'].values[tmin:tmax], target[f'{sol:s}_{dims[1]}'].values[tmin:tmax], '.-',
                                            linewidth=1, markersize=4, label=f'{sol:s} ({rmse:.0f})')
                        utils.labels(axs[a], dims[0], dims[1], f'{sc:s} / {ep:d} / {tar:d} ({cls:s})', 16)
                        axs[a].legend(fontsize=10)
                        a += 1
                    else:
                        sol = solvers[0]
                        axs[a].plot(target[f'{sol:s}_{dims[0]}'].values[tmin:tmax], target[f'{sol:s}_{dims[1]}'].values[tmin:tmax], '.-',
                                    linewidth=1, markersize=4, label=f'{tar:d}')
                if len(solvers) == 1:
                    utils.labels(axs[a], dims[0], dims[1], f'{sc:s} / {ep:d} / ({targets[sc][ep][0]:d}-{targets[sc][ep][-1]:d})', 16)
                    axs[a].legend(fontsize=10)
                    a += 1

        return axs

    def plot_trajectories_3d(self, scenarios=None, episodes=None, targets=None, n_episodes=2, n_targets=2,
                             solvers=None, add_naive=True, group=None, figsize=(5,4), axs=None):
        if scenarios is None: scenarios = self.scenarios
        if solvers is None: solvers = self.default_solvers_order
        if add_naive and ('Naive' in self.default_solvers_order) and ('Naive' not in solvers): solvers = ['Naive'] + solvers
        if episodes is None:
            episodes = {sc: np.arange(min(n_episodes, len(self.targets[sc]))) for sc in scenarios}
        if not isinstance(episodes, dict):
            episodes = {sc:episodes for sc in scenarios}
        if targets is None:
            targets = {sc:{ep:np.arange(min(n_targets, len(self.targets[sc][ep]))) for ep in episodes[sc]} for sc in scenarios}
        if not isinstance(targets, dict):
            targets = {sc:{ep:targets for ep in episodes[sc]} for sc in scenarios}
        if axs is None: axs = []
        filt = None if group is None else GroupFilter(self.meta_episodes, group)

        a = 0
        for sc in scenarios:
            for ep in episodes[sc]:
                if filt is not None and not filt.isin(sc,ep):
                    continue
                for tar in targets[sc][ep]:
                    cls = self.meta_targets[(self.meta_targets.scenario==sc)&(self.meta_targets.episode==ep)&(self.meta_targets.target==tar)].target_class.values[0]
                    target = self.targets[sc][ep][tar]
                    plt.figure(figsize=figsize)
                    ax = plt.axes(projection='3d')
                    # plot target
                    ax.plot3D(target.x_x, target.x_y, target.x_z, 'k.-', linewidth=1, markersize=3)
                    ax.plot3D([target.x_x.values[ 0]], [target.x_y.values[ 0]], [target.x_z.values[ 0]], 'k>', markersize=8)
                    ax.plot3D([target.x_x.values[-1]], [target.x_y.values[-1]], [target.x_z.values[-1]], 'ks', markersize=8)
                    # plot estimations
                    for sol in solvers:
                        rmse = np.sqrt(target[f'SE_{sol:s}'].mean())
                        ax.plot3D(target[f'{sol:s}_x'], target[f'{sol:s}_y'], target[f'{sol:s}_z'], '.-',
                                    linewidth=1, markersize=3, label=f'{sol:s} ({rmse:.0f})')
                    utils.labels(ax, None, None, f'{sc:s} / {ep:d} / {tar:d} ({cls:s})', 16)
                    ax.legend(fontsize=10)
                    axs.append(ax)
                    a += 1

        return axs

    def plot_vertical_trajectory(self, sc=None, ep=0, tar=0, solvers=None, ax=None):
        if sc is None: sc = self.scenarios[0]
        if solvers is None: solvers = self.default_solvers_order
        if ax is None: ax = utils.Axes(1,1,axsize=(8,4))[0]
        cls = self.meta_targets[(self.meta_targets.scenario==sc)&(self.meta_targets.episode==ep)&(self.meta_targets.target==tar)].target_class.values[0]

        pd.options.mode.chained_assignment = None  # default='warn'
        r = self.res[self.res.identifier==f'{sc}_{ep}_{tar}']
        r = r[r.solver.isin(solvers)]
        r0 = r[r.solver==solvers[0]]
        r0['solver'] = 'Truth'
        r0['z'] = r0['x_z']
        r = pd.concat((r0,r))
        pd.options.mode.chained_assignment = 'warn'

        sns.pointplot(data=r, x='t', y='z', hue='solver', scale=0.3, ax=ax)
        utils.labels(ax, None, None, f'{sc:s} / {ep:d} / {tar:d} ({cls:s})', 16)

        return ax

    def runtime_analysis(self, solvers=None, axs=None, show_outliers=True):
        if solvers is None: solvers = self.default_solvers_order
        if axs is None: axs = utils.Axes(2,2,axsize=(6,4))

        dd = pd.DataFrame()
        for sol_nm in solvers:
            solver = self.solvers[sol_nm]
            for module, times in solver.runtime_record.items():
                dd = pd.concat((dd, pd.DataFrame(dict(
                    solver = len(times) * [sol_nm],
                    module = len(times) * [module],
                    runtime = times,
                ))))
        dd.reset_index(drop=True, inplace=True)
        sns.boxplot(data=dd, x='module', hue='solver', y='runtime', showmeans=True, showfliers=show_outliers, ax=axs[0])
        utils.labels(axs[0], 'Module', 'Runtime [s]', fontsize=16)
        axs[0].legend(fontsize=12)

        try:
            dd = pd.DataFrame()
            for sol_nm in solvers:
                solver = self.solvers[sol_nm]
                times = 0
                for module, ts in solver.runtime_record.items():
                    times += np.array(ts)
                dd = pd.concat((dd, pd.DataFrame(dict(
                    solver = len(times) * [sol_nm],
                    quantile = 100*np.arange(len(times))/(len(times)-1),
                    runtime = sorted(times),
                ))))
            dd.reset_index(drop=True, inplace=True)
            sns.lineplot(data=dd, x='quantile', hue='solver', y='runtime', ax=axs[1], ci=None)
            utils.labels(axs[1], 'Quantile [%]', 'Runtime [s]', fontsize=16)
            axs[1].legend(fontsize=12)
            axs[1].set_yscale('log')
        except:
            pass

        return axs

    def compare_solvers(self, sol0=None, solvers=None, group=None, goal='RMSE', goal_units='', target_classes=None,
                        confidence_interval=95, plot_range=95, detailed=False, axs=None, fontsize=14):
        if sol0 is None: sol0 = self.default_solvers_order[0]
        if solvers is None: solvers = self.default_solvers_order
        solvers = [s for s in solvers if s!=sol0]
        if axs is None: axs = utils.Axes(detailed*len(solvers)+4, 3, axsize=(6,4))
        a = 0

        rr = self.meta_targets.copy()
        if group is not None:
            rr = rr[rr.group==group]
        if target_classes is None:
            target_classes = sorted(np.unique(rr.target_class))

        if goal == 'MSE' and f'MSE_{sol0:s}' not in rr.columns:
            rr[f'MSE_{sol0:s}'] = rr[f'RMSE_{sol0:s}']**2
            for sol2 in solvers:
                rr[f'MSE_{sol2:s}'] = rr[f'RMSE_{sol2:s}']**2

        for sol2 in solvers:
            rr['delta'] = rr[f'{goal:s}_{sol2:s}'] - rr[f'{goal:s}_{sol0:s}']
            rr[f'{sol2:s}'] = rr.delta
            if not detailed:
                continue
            ax = axs[a]
            a += 1
            for cls in target_classes:
                r = rr[rr.target_class==cls]
                n = len(r)
                m = r.delta.mean()
                s = r.delta.std()
                z = f'{m/s*np.sqrt(n):.2f}' if s>0 else 'nan'
                lab = f'{cls:s} (z={m:.1f}/{s:.1f}*sqrt({n:d})={z:s})'
                utils.plot_quantiles(r.delta.values, ax=ax, showmeans=True, label=lab)
            ax.legend(fontsize=13)
            if goal_units:
                ylab = f'{goal:s} [{goal_units:s}]\n{sol2:s} - {sol0:s}'
            else:
                ylab = f'{goal:s}: {sol2:s} - {sol0:s}'
            utils.labels(ax, 'target quantile [%]', ylab, fontsize=fontsize)
            lm = max(np.abs(rr.delta.quantile(1-plot_range/100)), np.abs(rr.delta.quantile(plot_range/100)))
            if lm > 0:
                ax.set_ylim((-lm,lm))
            # utils.fontsize(ax,fontsize,12,12)

        r = utils.pd_merge_cols(rr,solvers,'delta','solver')
        r['delta_w'] = r['delta'] * r['T'] / r['T'].mean()

        for i in range(4):
            ax = axs[a]
            sns.barplot(data=r, x=['target_class','solver'][i%2], hue=['solver','target_class'][i%2], y=['delta','delta_w'][i//2],
                        ax=ax, order=[target_classes,None][i%2], hue_order=[None,target_classes][i%2],
                        ci=confidence_interval, capsize=.1)
            ax.legend(fontsize=12)
            ylab = f'{goal:s} difference per target' if i<2 else f'{goal:s} diff. per target (weighted)'
            if goal_units: ylab = f'{ylab:s} [{goal_units:s}]'
            utils.labels(ax, ['target_class','solver'][i%2], ylab, f'Alternative solver vs. {sol0:s}', fontsize=fontsize)
            utils.fontsize(ax,fontsize,12,12)
            if len(target_classes) > 5:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
            a += 1

        plt.tight_layout()
        return axs

    def get_target_story(self, sc='test', ep=0, tar=0):
        target = self.targets[sc][ep][tar]
        args = self.target_args[sc][ep][tar]
        intervals = args['intervals']
        general = dict(max_acc=args['acc'], time=args['T'])

        phase = np.arange(len(intervals))
        mode = ['straight' if interval[0].__name__=='line' else 'turn' for interval in intervals]
        time = [(target.phase==i).sum() for i in range(len(intervals))]
        acc = [intr[1]['acc'] for intr in intervals]
        direction = [[['left','right'],['down','up']][intr[1]['vertical']][intr[1]['angle']>=0] \
                         if mode[i]=='turn' else 'forward' for i,intr in enumerate(intervals)]
        angle = [intr[1]['angle'] if 'angle' in intr[1] else 0 for intr in intervals]
        story = pd.DataFrame(dict(phase=phase, mode=mode, time=time, acc=acc, direct=direction, angle=angle))

        return story, general

    def target_analysis(self, sc=None, ep=0, tar=0, solvers=None, ref_solver=None, tmin=None, tmax=None):
        if sc is None: sc = self.scenarios[0]
        if solvers is None: solvers = self.default_solvers_order
        if ref_solver is None: ref_solver = solvers[0]

        print(self.get_target_story(sc,ep,tar)[0])

        axs0 = []
        axs0.append(self.plot_trajectories(scenarios=[sc], episodes=[ep], targets=[tar], solvers=solvers, group=1,
                                           dims=('x','y'), tmin=tmin, tmax=tmax, axs_args=dict(W=1, axsize=(10,8))))
        axs0.append(self.plot_trajectories(scenarios=[sc], episodes=[ep], targets=[tar], solvers=solvers, group=1,
                                           dims=('x','z'), tmin=tmin, tmax=tmax, axs_args=dict(W=1, axsize=(10,8))))

        self.plot_trajectories_3d(scenarios=[sc], episodes=[ep], targets=[tar], solvers=solvers, group=1, figsize=(10,8))

        target = self.targets[sc][ep][tar]
        x = target[f'x_x'].values
        y = target[f'x_y'].values
        z = target[f'x_z'].values
        cm = plt.cm.get_cmap('coolwarm')
        sol1 = ref_solver
        se1 = target[f'SE_{sol1}']
        sols2 = solvers[1:]
        axs = utils.Axes(2*len(sols2),2,axsize=(8,6),fontsize=15)
        for a,sol2 in enumerate(sols2):
            se2 = target[f'SE_{sol2}']
            score = (np.sqrt(se1) - np.sqrt(se2)).values
            ids = target.phase.diff().values != 0
            if tmin is not None: ids = ids & (np.arange(len(ids))>=tmin)
            if tmax is not None: ids = ids & (np.arange(len(ids))< tmax)
            ids[0] = True
            for i,ylab in enumerate(('y','z')):
                yy = y if ylab=='y' else z
                ax = axs[2*a+i]
                ax.plot(x[tmin:tmax], yy[tmin:tmax], 'k-', zorder=0)
                ax.plot(x[ids], yy[ids], 'ko', markersize=20, markerfacecolor='none')
                ax.scatter(x[tmin:tmax], yy[tmin:tmax], s=40, c=score[tmin:tmax], cmap=cm, vmin=-30, vmax=30, zorder=2)
                if f'Naive_x' in target.columns:
                    ax.plot(target[f'Naive_x'].values[tmin:tmax], target[f'Naive_{ylab}'].values[tmin:tmax], 'y.-', linewidth=1, zorder=1)
                axs.labs(2*a+i, 'x', ylab, f'{sol1} (blue) vs. {sol2} (red)')

        axs = utils.Axes(4,2,axsize=(7,3.5),fontsize=15)
        tt = self.res[self.res.identifier==f'{sc}_{ep:d}_{tar:d}']
        tt = tt[tt.solver.isin(solvers)]
        for i,dm in enumerate(('x','y','z')):
            utils.compare_quantiles(tt, 't', f'error_{dm}', 'solver', xbins=10, hbins=10, axs=[axs[i]])
        sns.boxplot(data=tt, hue='solver', x='phase', y='error', showmeans=True, ax=axs[3])
        axs.labs(3, 'phase (duration)', 'error')
        axs[3].set_xticklabels(tt.groupby('phase').apply(lambda d:f'{d.phase_class.values[0]} ({len(np.unique(d.t)):d})'), fontsize=13)
        plt.tight_layout()

        return axs0

    def targets_consistency_test(self, axs=None):
        if axs is None: axs = utils.Axes(2,2,axsize=(6,4))

        scenarios = []
        episodes = []
        tar_id = []
        phases = []
        dims = []
        errs = []
        errs_normalized = []
        for sc in self.targets:
            for ep in range(len(self.targets[sc])):
                for tar in range(len(self.targets[sc][ep])):
                    target = self.targets[sc][ep][tar]
                    for dm in ('x','y','z'):
                        v = target[f'x_v{dm}'].values
                        v = 0.5*(v[:-1]+v[1:])
                        err = np.abs( target[f'x_{dm}'].diff().values[1:] - v )
                        errs.extend(list(err))
                        errs_normalized.extend(list(100 * err / np.abs(target[f'x_v{dm}'].values[:-1])))
                        dims.extend(len(err)*[dm])
                        scenarios.extend(len(err)*[sc])
                        episodes.extend(len(err)*[ep])
                        tar_id.extend(len(err)*[tar])
                        phases.extend(target.phase.values[1:])
                    v = np.sqrt(target.x_vx**2+target.x_vy**2+target.x_vz**2).values
                    v = 0.5*(v[:-1]+v[1:])
                    dx = np.sqrt(target.x_x.diff()**2+target.x_y.diff()**2+target.x_z.diff()**2).values[1:]
                    err = np.abs(dx-v)
                    errs.extend(list(err))
                    errs_normalized.extend(list(100*err/np.abs(v)))
                    dims.extend(len(err)*['abs'])
                    scenarios.extend(len(err)*[sc])
                    episodes.extend(len(err)*[ep])
                    tar_id.extend(len(err)*[tar])
                    phases.extend(target.phase.values[1:])

        dd = pd.DataFrame(dict(scenario=scenarios, episode=episodes, target=tar_id, phase=phases, dimension=dims,
                               error=errs, error_rel=errs_normalized))
        sns.boxplot(data=dd, x='scenario', hue='dimension', y='error', ax=axs[0], showmeans=True)
        utils.labels(axs[0], 'scenario', 'location/speed inconsistency [m]', fontsize=15)
        sns.boxplot(data=dd, x='scenario', hue='dimension', y='error_rel', ax=axs[1], showmeans=True)
        utils.labels(axs[1], 'scenario', 'location/speed inconsistency\n[% of speed]', fontsize=15)
        axs[1].set_yscale('log')
        plt.tight_layout()
        return dd, axs

class GroupFilter:
    def __init__(self, meta_df, group):
        self.complete_episodes = not 'target' in meta_df.columns
        if self.complete_episodes:
            self.ref = [(sc,ep) for sc,ep,grp in \
                        zip(meta_df.scenario, meta_df.episode, meta_df.group) \
                        if grp == group]
        else:
            self.ref = [(sc,ep,tar) for sc,ep,tar,grp in \
                        zip(meta_df.scenario, meta_df.episode, meta_df.target, meta_df.group) \
                        if grp == group]
        self.ref = set(self.ref)

    def isin(self, sc, ep, tar=None):
        if self.complete_episodes:
            return (sc,ep) in self.ref
        return (sc,ep,tar) in self.ref

