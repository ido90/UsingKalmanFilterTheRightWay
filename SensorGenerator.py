'''
A simulation of a Doppler radar.

Written by Ido Greenberg, 2021
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import utils

class Radar:

    def __init__(self, noise_factor=1, noise_base=(1/180*3.14,3/180*3.14,10,5), cart_noise_base=(100,100,100,5),
                 polar=True, FAR=0, FN=0):
        self.polar = polar
        if not polar:
            noise_base = cart_noise_base

        self.noise = dict(
            theta = noise_factor*noise_base[0],
            phi   = noise_factor*noise_base[1],
            r     = noise_factor*noise_base[2],
            dop   = noise_factor*noise_base[3],
        )

        self.FAR = FAR
        self.FN = FN

        self.false_detection_range = dict(
            theta = (0,2*np.pi),
            Rxy = (0,5000),
            z = (-100,1000),
            dop = (-30,30),
        )

    def simulate_detection(self, target, noise=None):
        if noise is None:
            noise = self.noise

        dop = calc_doppler(target['x'], target['v'])
        dop = dop + noise['dop'] * np.random.randn(len(dop))

        if self.polar:
            r, theta, phi = cart2polar(*target['x'])
            theta = theta + noise['theta'] * np.random.randn(theta.size)
            phi = phi + noise['phi'] * np.random.randn(phi.size)
            r = r + noise['r'] * np.random.randn(r.size)
            x, y, z = polar2cart(r, theta, phi)
        else:
            n = target['x'].shape[1]
            x = target['x'][0] + np.random.randn(n) * noise['theta']
            y = target['x'][1] + np.random.randn(n) * noise['phi']
            z = target['x'][2] + np.random.randn(n) * noise['r']

        return x, y, z, dop

    def make_target_detections(self, target, noise=None, FN=None):
        if FN is None:
            FN = self.FN

        target_dict = dict(
            t = target.t.values,
            x = target.loc[:, ('x_x','x_y','x_z')].values.transpose(),
            v = target.loc[:, ('x_vx','x_vy','x_vz')].values.transpose(),
        )
        x, y, z, dop = self.simulate_detection(target_dict, noise)
        dd = pd.DataFrame(dict(
            t = target.t.values,
            z_x = x,
            z_y = y,
            z_z = z,
            z_dop = dop
        ))

        # false negatives (i.e. missing detections) - only from the second time-step
        if FN:
            ids = 1 + np.where(np.random.random(len(dd)-1) < FN)[0]
            dd.iloc[ids, 1:] = np.nan

        return dd

    def make_false_detection(self, ranges=None):
        if ranges is None:
            ranges = self.false_detection_range

        r1, r2 = ranges['Rxy']
        theta1, theta2 = ranges['theta']
        z1, z2 = ranges['z']
        dop1, dop2 = ranges['dop']

        r = r1 + (r2-r1) * np.sqrt(np.random.random())
        theta = theta1 + (theta2-theta1) * np.random.random()
        z = z1 + (z2-z1) * np.random.random()
        dop = dop1 + (dop2-dop1) * np.random.random()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z, dop

    def make_false_detections(self, ti, tf, dt, FAR=None, range_args=None):
        if FAR is None:
            FAR = self.FAR
        if FAR == 0:
            return pd.DataFrame()

        times = np.arange(ti, tf, dt).astype(int)
        n_false = np.round(np.random.exponential(FAR, len(times))).astype(int)

        if np.sum(n_false) == 0:
            return pd.DataFrame()

        tt, x, y, z, dop = [[] for _ in range(5)]
        for i,t in enumerate(times):
            for _ in range(n_false[i]):
                tt.append(t)
                xx, yy, zz, ddop = self.make_false_detection(range_args)
                x.append(xx)
                y.append(yy)
                z.append(zz)
                dop.append(ddop)

        dd = pd.DataFrame(dict(
            t = tt,
            z_x = x,
            z_y = y,
            z_z = z,
            z_dop = dop,
            target = len(tt) * [-1]
        ))

        return dd

    def make_episode_detections(self, targets, noise=None, FA_args=None, FAR=None, dt=None, update_targets=False):
        if dt is None: dt = targets[0].t.values[1] - targets[0].t.values[0]
        Ti = np.min([tar.t.values[ 0] for tar in targets])
        Tf = np.max([tar.t.values[-1] for tar in targets])

        dd = pd.DataFrame()
        for i, tar in enumerate(targets):
            d = self.make_target_detections(tar, noise)
            d['target'] = len(d) * [i]
            if update_targets:
                for c in d.columns:
                    if c not in ('t','target'):
                        targets[i][c] = d[c].values
            dd = pd.concat((dd, d))

        dd = pd.concat((dd, self.make_false_detections(Ti, Tf, dt, FAR, FA_args)))

        dd.sort_values('t', inplace=True)
        dd.reset_index(drop=True, inplace=True)

        return dd


def cart2polar(x, y, z):
    R = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / R)
    return R, theta, phi

def polar2cart(r, theta, phi):
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return x, y, z

def calc_doppler(x, v):
    return np.sum(x*v, axis=0) / np.linalg.norm(x, axis=0)

def show_detections(dd, by='target', ax=None, title=None, max_targets=6, show_fa=True):
    # by = either 'target' or 't'
    if ax is None:
        ax = utils.Axes(1)[0]

    dd = dd[dd.target < max_targets]
    if not show_fa:
        dd = dd[dd.target >= 0]

    if by == 'target':
        palette = sns.color_palette(n_colors=len(np.unique(dd.target)))
        sns.scatterplot(data=dd, x='z_x', y='z_y', hue='target', ax=ax,
                        palette=palette)
    elif by =='t':
        sns.scatterplot(data=dd, x='z_x', y='z_y', hue='t', ax=ax)

    utils.labels(ax, 'x', 'y', title, 16)
    return ax

def show_detections_vs_targets(targets, ax=None, title=None, max_targets=6):
    if ax is None:
        ax = utils.Axes(1)[0]

    for i, tar in enumerate(targets[:max_targets]):
        h = ax.plot(tar.x_x.values, tar.x_y.values, '-', linewidth=1, label=i)[0]
        ax.plot(tar.z_x.values, tar.z_y.values, '.', markersize=5, color=h.get_color())

    utils.labels(ax, 'x', 'y', title, 16)
    ax.legend(fontsize=10)

    return ax

def detections_summary(rr, targets, axs=None, tit=''):
    if axs is None: axs = utils.Axes(4, 4, axsize=(4, 3))
    show_detections(rr, ax=axs[0])
    show_detections(rr, 't', ax=axs[1])
    show_detections(rr, 't', ax=axs[2], show_fa=False)
    show_detections_vs_targets(targets, ax=axs[3])
    if tit:
        utils.labels(axs[0], title=tit, fontsize=14)
    plt.tight_layout()
    return axs
