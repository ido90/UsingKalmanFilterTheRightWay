'''
Written by Ido Greenberg, 2020
'''

import numpy as np
from scipy import stats
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from warnings import warn
import pickle as pkl
from time import time

import utils

class Solver:
    def __init__(self, matcher=None, matcher_args=None, tracker=None, tracker_args=None, dt=1):
        if matcher is None: matcher = HungarianAssigner
        if matcher_args is None: matcher_args = {}
        if tracker is None: tracker = KF_default
        if tracker_args is None: tracker_args = {}
        tracker_args = utils.update_dict(tracker_args, dict(dt=dt))

        self.matcher_fun = matcher
        self.matcher_args = matcher_args
        self.matcher = self.matcher_fun(**self.matcher_args)

        self.tracker_fun = tracker
        self.tracker_args = tracker_args
        self.trackers = []
        self.killed_trackers = []
        self.n_tracks = 0
        self.global_tracks_index_base = 0

        self.targets_record = dict(track=[], target=[], nll=[])
        self.runtime_record = dict(predict=[], update=[])

    def initialize(self, reset_records=False):
        self.trackers = []
        self.killed_trackers = []
        self.global_tracks_index_base += self.n_tracks
        self.n_tracks = 0
        if reset_records:
            self.global_tracks_index_base = 0
            self.targets_record = dict(track=[], target=[], nll=[])
            self.runtime_record = dict(predict=[], update=[])

    def get_tracker_by_id(self, i):
        for tr in self.trackers:
            if tr.target_id == i:
                return tr
        return None

    def predict(self):
        for trk in self.trackers:
            t0 = time()
            trk.predict()
            self.runtime_record['predict'].append(time()-t0)

    def update(self, detections, matches, lonely_detections, lonely_trackers, targets):
        # matches
        for d, t in matches:
            t0 = time()
            self.trackers[t].update(detections[d])
            self.runtime_record['update'].append(time()-t0)
            self.targets_record['track'].append(self.global_tracks_index_base + self.trackers[t].target_id)
            self.targets_record['target'].append(targets[d])
            self.targets_record['nll'].append(self.trackers[t].nlls[-1])

        # non-continued tracks - kill if needed
        killed = []
        for t in lonely_trackers:
            self.trackers[t].update(None)
            if self.trackers[t].killed:
                killed.append(t)
        killed = sorted(killed, reverse=True)
        for t in killed:
            self.killed_trackers.append(self.trackers.pop(t))

        # un-assigned detections - create tracker
        for d in lonely_detections:
            trk = self.tracker_fun(target_id=self.n_tracks, **self.tracker_args)
            trk.update(detections[d])
            self.trackers.append(trk)
            self.targets_record['track'].append(self.global_tracks_index_base + trk.target_id)
            self.targets_record['target'].append(targets[d])
            self.targets_record['nll'].append(np.nan)
            self.n_tracks += 1

    def get_assignments(self, matches, targets, new_assignment_code=None):
        n = len(targets)
        if matches:
            matches = np.array(matches)
            dets = matches[:,0]
            tracks = matches[:,1]
        else:
            dets = []

        assigns = []
        unassigned_count = 0
        for d in range(n):
            if d in dets:
                i = np.where(dets==d)[0][0]
                assigns.append(self.trackers[tracks[i]].target_id)
            else:
                if new_assignment_code:
                    assigns.append(new_assignment_code)
                else:
                    assigns.append(self.n_tracks+unassigned_count) # targets[d])
                    unassigned_count += 1

        return assigns

    def step(self, detections=tuple(), targets=tuple()):
        self.predict()
        matches, lonely_dets, lonely_trks = self.matcher.assign(self.trackers, detections)
        assigns = self.get_assignments(matches, targets)
        self.update(detections, matches, lonely_dets, lonely_trks, targets)
        return assigns

    # get_stats(): activated, killed, noise=killed&(not activated), killed lengths distribution...

######################   MATCHERS   ######################

class Matcher:
    def __init__(self):
        pass
    def assign(self, trackers, dets):
        raise NotImplementedError()

class HungarianAssigner(Matcher):
    def __init__(self, match_thresh=np.inf, nll_thresh_mode=True):
        super(HungarianAssigner, self).__init__()
        if not nll_thresh_mode:
            match_thresh = -np.log(match_thresh) if match_thresh>0 else np.inf
        self.match_thresh = match_thresh

    def assign(self, trackers, dets, match_thresh=None, nll_thresh_mode=False):
        # empty case
        if len(dets)==0 or len(trackers)==0:
            return [], np.arange(len(dets)), np.arange(len(trackers))

        # input pre-processing
        if match_thresh is None:
            match_thresh = self.match_thresh
        elif not nll_thresh_mode:
            match_thresh = -np.log(match_thresh)

        # calculate likelihoods
        NLL = np.zeros((len(dets), len(trackers)), dtype=np.float32)
        for d, det in enumerate(dets):
            dpos = det[1:]
            for t, trk in enumerate(trackers):
                NLL[d,t] = trk.nll(dpos)

        # find best match
        det_i, trk_i = linear_sum_assignment(NLL)

        # find unmatched entries
        lonely_dets = [d for d in range(len(dets)) if d not in det_i]
        lonely_trks = [t for t in range(len(trackers)) if t not in trk_i]

        # don't count low-likelihood matches
        matches = []
        for di, ti in zip(det_i, trk_i):
            if NLL[di,ti] > match_thresh:
                lonely_dets.append(di)
                lonely_trks.append(ti)
            else:
                matches.append([di,ti])

        return matches, lonely_dets, lonely_trks

######################   TRACKERS   ######################

class Tracker:
    trackers_count = 0

    def __init__(self, target_id=None, dt=1, state_dim=6, obs_dim=4, declare_after=2, kill_after=1):
        self.id = Tracker.trackers_count
        Tracker.trackers_count += 1

        self.dt = dt
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.declare_after = declare_after
        self.kill_after = kill_after

        self.x = np.array(self.state_dim*[np.nan])
        self.x_history = []
        self.detection_indicator = []
        self.time_step = 0
        self.idle_count = 0
        self.dets_count = 0
        self.nlls = []
        self.declared = False
        self.killed = False
        self.target_id = target_id

    def initialize(self, target_id=None):
        self.x = np.array(self.state_dim*[np.nan])
        self.x_history = []
        self.detection_indicator = []
        self.time_step = 0
        self.idle_count = 0
        self.dets_count = 0
        self.nlls = []
        self.declared = False
        self.killed = False
        self.target_id = target_id

    def predict(self):
        pass

    def update(self, detection):
        '''
        :param detection: [t, z_x, z_y, z_z, z_dop]
        '''
        self.time_step += 1
        self.detection_indicator.append(detection is not None)
        self.x_history.append(self.x)

        if detection is None:
            self.idle_count += 1
            if self.idle_count >= self.kill_after:
                self.killed = True
            return

        self.nlls.append(self.nll(detection[1:]))
        self.idle_count = 0
        self.dets_count += 1
        if not self.declared and (self.dets_count >= self.declare_after):
            self.declared = True

        return self.do_update(detection)

    def do_update(self, det):
        raise NotImplementedError()
    def get_pos(self):
        return np.array(self.x[:3])
    def nll(self, det):
        if det is None:
            return 0
        if np.isnan(self.get_pos()[0]):
            return 0
        return self.calc_nll(det)
    def calc_nll(self, det):
        return np.sum(np.array(det[:3]-self.get_pos())**2)
    def likelihood(self, det):
        return np.exp(-self.nll(det))

class NaiveTracker(Tracker):
    def __init__(self, **kwargs):
        super(NaiveTracker, self).__init__(state_dim=3, **kwargs)
    def do_update(self, det):
        self.x = det[1:4]

class KF_default(Tracker):
    def __init__(self, P_factor=1e3, Q_factor=2, R_factor=20, use_Jac=False, **kwargs):
        super(KF_default, self).__init__(**kwargs)
        self.use_Jac = use_Jac
        self.f = None
        self.P_factor = P_factor
        self.Q_factor = Q_factor
        self.R_factor = R_factor
        self.reset()
    def reset(self):
        self.f = KalmanFilter(dim_x=self.state_dim, dim_z=self.obs_dim)
        self.f.x = np.array(self.state_dim*[np.nan])
        self.set_pred_matrix()
        self.f.H = None
        self.f.P *= self.P_factor**2
        self.f.Q *= self.Q_factor**2
        self.f.R *= self.R_factor**2
    def set_pred_matrix(self):
        dt = self.dt
        self.f.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]]
        )
    def set_transform_matrix(self, det, use_Jac=None, eps=1e-3):
        if use_Jac is None:
            use_Jac = self.use_Jac
        if use_Jac:
            # EKF derivation
            x, y, z = self.x[:3]
            vx, vy, vz = self.x[3:]
            # avoid zeros for numeric stability
            x = x if np.abs(x)>=eps else (eps if x>=0 else -eps)
            y = y if np.abs(y)>=eps else (eps if y>=0 else -eps)
            z = z if np.abs(z)>=eps else (eps if z>=0 else -eps)
            vx = vx if np.abs(vx)>=eps else (eps if vx>=0 else -eps)
            vy = vy if np.abs(vy)>=eps else (eps if vy>=0 else -eps)
            vz = vz if np.abs(vz)>=eps else (eps if vz>=0 else -eps)

            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            p = x*vx+y*vy+z*vz
            self.f.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [vx/r-p*x/r**3, vy/r-p*y/r**3, vz/r-p*z/r**3, x/r, y/r, z/r]
            ])
        else:
            # KF: use x=z=detection instead of x=self.x (presumably more accurate)
            x, y, z = det
            # avoid zeros for numeric stability
            x = x if np.abs(x)>=eps else (eps if x>=0 else -eps)
            y = y if np.abs(y)>=eps else (eps if y>=0 else -eps)
            z = z if np.abs(z)>=eps else (eps if z>=0 else -eps)

            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            self.f.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, x/r, y/r, z/r]
            ])
    def predict(self):
        self.f.predict()
        self.x = self.f.x
    def do_update(self, det):
        self.set_transform_matrix(det[1:4])
        if np.isnan(self.x[0]):
            self.f.x = np.array(list(det[1:4]) + [0,0,0])
            self.f.S = np.dot(self.f.H,np.dot(self.f.P,self.f.H.transpose())) + self.f.R
        else:
            self.f.update(det[1:5])
        self.x = self.f.x
    def calc_nll(self, det):
        self.set_transform_matrix(det[:3])
        return -self.f.log_likelihood_of(det[:4])
