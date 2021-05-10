'''
Written by Ido Greenberg, 2020

_______________________________

Tips & tricks for dealing with numeric instability (nans) in gradients:
    use torch.autograd.set_detect_anomaly(True)
    dividing by 0
    1/norm(eps)
    very large losses
    adam.eps
    sgd instead of adam
    log-sum-exp trick
https://discuss.pytorch.org/t/solved-debugging-nans-in-gradients/10532/6
'''

import numpy as np
from warnings import warn
import pickle as pkl
from time import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mp

import utils
import Trackers


class NeuralKF(nn.Module, Trackers.Tracker):
    '''
    This can be seen as a combination of [a Neural extension of KF] and [particle filters].
    The "particle filters" approach is expressed by having multiple parallel KFs.
    The Neural extension incorporates a prediction of the acceleration into the KF.
    In addition, some of KF parameters (Q,R) are learned rather than manually-defined.

    As in KF, we run predict & update steps alternately:
    - Predict:
        - Neural prediction (neural component): predict acceleration using a multi-layer LSTM
                                                with state (x,v) and observations (z) as an input.
                                                apply the prediction n times with different weights
                                                to generate n estimates of the acceleration ("split to particles").
        - Standard prediction (as in KF with known acceleration): for each estimate of a:
                                                                  progress time by dt: x,v,a -> x,v
    - Update:
        - Select (filter component): select the state prediction that fits the new observation best ("merge particles").
        - Update (as in KF): merge info from state & observation.
    '''

    def __init__(self, n_gaussians=1, dt=1, d_extractor=(2,48), d_hidden=(2,64), dropout=0.0, precision=torch.float64,
                 Pfac=1e3, Qfac=2, Rfac=20, update_loss_fac=None, seed_base=0, seed=1, ML_estimator=True, loss_p=2,
                 a_skip_conn=False, merge_on_update=False, separate_hidden_states=None, aggregate_prob=True, update_iters=1,
                 H_from_obs=True, normalize_features=True, radial_acc=True, naive_tracker=False, basic_KF=False, init_nan=True,
                 no_acc=False, pred_acc=True, const_Q=True, const_R=True, no_corr=False, estimate_R=True, estimate_Q=True, polar_R=False, cheat_R=False,
                 full_nll_train=True, full_nll_eval=True, multi_Q=False, dynamic_Q=False, head_layer=False, pred_H=0, dynamic_R=False,
                 feed_t=True, feed_P=False, feed_Q=None, optimize_per_target=False, estimate_from_valid=True,
                 manual_head_clip=False, exp_cov_transform=True, input_obs=1, rotate_input=True, EKF=False, estimate_R_from_HZ=False,
                 device='cpu', record=None, load=False, title='NKF', **kwargs):
        # super(NeuralKF, self).__init__()
        nn.Module.__init__(self)
        Trackers.Tracker.__init__(self, **kwargs)
        if naive_tracker or basic_KF:
            n_gaussians = 1
            no_acc = True
            const_Q = True
            const_R = True
            no_corr = True
        self.seed = seed_base + seed
        self.set_seed()
        self.device = 'cpu'
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                warn('cuda unavailable, using cpu.')
        self.title = title
        self.dt = dt
        self.naive_tracker = naive_tracker
        self.EKF = EKF
        self.ML_estimator = ML_estimator # maximum likelihood estimator (center of largest gaussian) or expected value (weighted mean of gaussians)
        self.a_skip = a_skip_conn        # skip connection for a: a=a+f rather than a=f.
        self.merge_on_update = merge_on_update
        self.separate_hidden_states = separate_hidden_states
        if self.separate_hidden_states is None:
            self.separate_hidden_states = not self.merge_on_update
        self.aggregate_p = aggregate_prob # estimate each gaussian's amplitude on top of the last amplitude
        self.precision = precision
        self.init_nan = init_nan         # initialize to nan (first update is x=z) or to 0 (first update is regular)
        self.input_obs = input_obs       # number of previous observations fed to the NN as input
        self.no_acc = no_acc             # keep zero accuracy (don't learn it)
        if dynamic_R: const_R = False
        self.const_Q = const_Q           # keep constant noise estimator Q (don't learn it - Q would not have gradient)
        self.const_R = const_R           # keep constant noise estimator R (don't learn it)
        self.no_corr = no_corr           # keep zero correlation in Q & R (don't learn it)
        self.estimate_from_valid = estimate_from_valid # if estimating params from data - whether to use train or train+valid
        self.use_R_estimation = estimate_R # whether to estimate the observation noise covariance R directly from the data, or learn it using GD
        self.use_Q_estimation = estimate_Q # whether to estimate the process noise covariance Q directly from the data, or learn it using GD
        self.estimated_R = False         # a flag noting whether R was already estimated (if we initialize R with estimation and then train it from there, we use this flag to avoid overwriting it with another estimation)
        self.estimated_Q = False         # a flag noting whether Q was already estimated (if we initialize Q with estimation and then train it from there, we use this flag to avoid overwriting it with another estimation)
        self.estimate_R_from_HZ = estimate_R_from_HZ # when estimating R according to the observation model H, use H(Z) rather than H(X)
        self.polar_R = polar_R           # treat R in polar coordinates
        self.cheat_R = cheat_R           # use the known observation noise matrix R
        self.dynamic_R = dynamic_R       # predict R dynamically
        self.H_from_obs = H_from_obs     # define observation model H from observed position (rather than modeled one)
        self.pred_H = pred_H             # whether to predict H
        self.update_iters = update_iters # number of consecutive updates per step (see Iterated EKF)
        self.pred_acc = pred_acc         # predict acceleration rather than position & velocity
        self.normalize_features = normalize_features # normalize NN input
        self.feed_t = feed_t             # use time-step as input to the NN
        self.feed_P = feed_P             # use uncertainty (P) as input to the NN
        if feed_Q is None: feed_Q = (dynamic_Q=='nn')
        self.feed_Q = feed_Q             # use Q as input to the NN
        self.rotate_input = rotate_input # rotate network input (x & z) according to current motion direction
        self.dropout = dropout           # currently not used anywhere
        self.radial_acc = radial_acc     # network's output will be acceleration in radial coordinates
        self.d_hidden = d_hidden         # LSTM's hidden layers dimensions
        self.Pfac = Pfac                 # P,Q,R are initialized to I by default. change their initial value here.
        self.Qfac = Qfac
        self.Rfac = Rfac
        self.exp_cov_transform = exp_cov_transform # represent positive variances as exp() rather than softplus()
        self.multi_Q = multi_Q or dynamic_Q # use different Q for each gaussian
        self.dynamic_Q = dynamic_Q or not self.pred_acc # predict Q every step
        self.frozen_Q_pred = False       # freeze Q-prediction (use estimated Q) to stabilize training
        self.manual_head_clip = manual_head_clip # use [tanh with manually-chosen scaling of 1e3] for network output instead of [linear]
        self.full_nll_train = full_nll_train # full likelihood calculation - including velocity - on training mode
        self.full_nll_eval = full_nll_eval # full likelihood calculation - including velocity - on eval mode
        self.optimize_per_target = optimize_per_target # calculate training loss as sum over targets rather than sum over detections (i.e. avoid overweighting longer trajectories)
        self.update_loss_p = loss_p      # train loss is location error in power of p
        self.update_loss_fac = update_loss_fac
        if self.update_loss_fac is None:
            self.update_loss_fac = 1/50 if self.update_loss_p==2 else 0.5
        self.record_history = record     # T/F = T/F; None = record iff on eval state
        self.ng = n_gaussians            # number of gaussians to model in parallel

        # explicit state variables:
        self.t = 0      # time in sequence
        self.base_g = 0 # the gaussian from which the prediction step is done
        self.x = None   # position & velocity (dim: number of gaussians in the mixture X 6)
        self.x_prev = None # post-update position & velocity; allows comparing predicted position to last estimated position
        self.P = None   # position & velocity covariance (dim: 6x6)
        self.S = None   # HPH+R
        self.p = None   # probability of any gaussian in the mixture
        # last observation
        self.z = None   # last observation
        # hidden state variables: n_gaussians x acc (radial & theta & phi) & other hidden variables
        d_out = 1 + (3 if self.pred_acc else 6) + (self.dynamic_Q=='lstm')*(6 + (not self.no_corr)*15) # p, a, Q
        self.a = None
        self.h = [[torch.zeros((1,d_hidden[1]), dtype=self.precision).to(self.device) for _ in range(d_hidden[0]-1)]
                  for _ in range(self.ng)] # LSTM hidden state
        self.c = [[torch.zeros((1,d_out if j==d_hidden[0]-1 else d_hidden[1]), dtype=self.precision).to(self.device)
                   for j in range(d_hidden[0])]
                  for _ in range(self.ng)] # LSTM memory cell
        # recorded history in a sequence
        self.recording = self.record_history # True=True, False=False, None=only_in_inference_mode
        self.record = {}
        self.init_state()

        # unlearned transition matrices
        self.F = torch.tensor([
            [1.,0,0,self.dt,0,0],
            [0,1,0,0,self.dt,0],
            [0,0,1,0,0,self.dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=self.precision).to(self.device)
        self.H = None
        # learned noise matrices
        self.Q = None
        self.R = None
        Q_diag_transform = torch.log if self.exp_cov_transform else (lambda x:x)
        if self.const_Q or self.dynamic_Q:
            self.Q_diag = [Q_diag_transform(self.Qfac * torch.ones(6, dtype=self.precision)).to(self.device) for _ in range(self.ng)]
        else:
            self.Q_diag = nn.ParameterList([nn.Parameter(Q_diag_transform(self.Qfac*(0.5+torch.rand(6, dtype=self.precision))).to(self.device), requires_grad=True)
                                            for _ in range(self.ng)])
        if self.no_corr or self.const_Q or self.dynamic_Q:
            self.Q_corr = [torch.zeros(5 * 6 // 2, dtype=self.precision).to(self.device) for _ in range(self.ng)]
        else:
            self.Q_corr = nn.ParameterList([nn.Parameter(self.Qfac/3*torch.randn(5*6//2, dtype=self.precision).to(self.device), requires_grad=True)
                                            for _ in range(self.ng)])

        if self.const_R or self.dynamic_R:
            self.R_diag = (self.Rfac * torch.ones(4, dtype=self.precision)).log().to(self.device)
        else:
            self.R_diag = nn.Parameter((self.Rfac*(0.5+torch.rand(4, dtype=self.precision))).log().to(self.device), requires_grad=True)
        if self.no_corr or self.const_R or self.dynamic_R:
            self.R_corr = torch.zeros(3 * 4 // 2, dtype=self.precision).to(self.device)
        else:
            self.R_corr = nn.Parameter(self.Rfac/3*torch.randn(3*4//2, dtype=self.precision).to(self.device), requires_grad=True)

        NN = self.ng>1 or not self.no_acc

        # feature extractor
        self.dim_features = 8 + (1 if self.feed_t else 0) + (6 if self.feed_P else 0) + (6 if self.feed_Q else 0)
        d_in = 4 + self.input_obs*6 + self.dim_features
        self.extractor = None
        if NN and d_extractor is not None:
            n_layers = d_extractor[0]
            layers_width = d_extractor[1]
            if isinstance(layers_width, int):
                layers_width = n_layers*[layers_width]
            extractor = []
            for i in range(n_layers):
                d_out_fe = layers_width[i]
                if self.precision == torch.double:
                    extractor.extend([nn.Linear(d_in, d_out_fe).to(self.device).double(), nn.ReLU().double()])
                else:
                    extractor.extend([nn.Linear(d_in, d_out_fe).to(self.device), nn.ReLU()])
                d_in = d_out_fe
            self.extractor = nn.Sequential(*extractor)

        # hidden state transitions
        self.layers = None
        if NN:
            self.layers = [[nn.LSTMCell(d_in if j==0 else d_hidden[1], d_out if j==d_hidden[0]-1 else d_hidden[1]).to(self.device)
                            for j in range(d_hidden[0])] for _ in range(self.ng)]
            if self.precision == torch.double:
                self.layers = [[layer.double() for layer in layers] for layers in self.layers]
            self.layers = nn.ModuleList([nn.ModuleList(l) for l in self.layers])

        # Q extracting network
        self.Qnet = None
        if NN and self.dynamic_Q=='nn':
            if self.precision == torch.double:
                self.Qnet = nn.Linear(d_hidden[1]+d_out, 6).to(self.device).double()
            else:
                self.Qnet = nn.Linear(d_hidden[1]+d_out, 6).to(self.device)

        # head layer
        self.head_layer = None
        if NN and head_layer:
            if self.precision == torch.double:
                self.head_layer = nn.Linear(d_out-1, d_out-1).to(self.device).double()
            else:
                self.head_layer = nn.Linear(d_out-1, d_out-1).to(self.device)

        # H extractor
        self.Hnet = None
        if self.pred_H > 0:
            # features: x (6), P (6), z (4)
            if self.precision == torch.double:
                self.Hnet = nn.Sequential(nn.Linear(6+6+4,64),nn.Tanh(),nn.Linear(64,3+3*(self.pred_H>1)),nn.Tanh()).to(self.device).double()
            else:
                self.Hnet = nn.Sequential(nn.Linear(6+6+4,64),nn.Tanh(),nn.Linear(64,3+3*(self.pred_H>1)),nn.Tanh()).to(self.device)

        # R extracting network
        self.Rnet = None
        if self.dynamic_R:
            if self.precision == torch.double:
                self.Rnet = nn.Sequential(nn.Linear(6+6+4,64),nn.Tanh(),nn.Linear(64,4)).to(self.device).double()
            else:
                self.Rnet = nn.Sequential(nn.Linear(6+6+4,64),nn.Tanh(),nn.Linear(64,4)).to(self.device)

        # initialize model
        if load:
            if not isinstance(load, str): load = None
            self.load_model(load)
        self.eval()

    def set_seed(self, seed=None, cuda=False):
        if seed is None: seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def reset_model(self):
        if self.Q_diag[0].requires_grad:
            Q_diag_transform = torch.log if self.exp_cov_transform else (lambda x:x)
            for i in range(self.ng):
                self.Q_diag[i] = nn.Parameter(Q_diag_transform(self.Qfac*(0.5+torch.rand(6, dtype=self.precision))).to(self.device), requires_grad=True)
        if self.R_diag.requires_grad:
            self.R_diag = nn.Parameter((self.Rfac*(0.5+torch.rand(4, dtype=self.precision))).log().to(self.device), requires_grad=True)
        if self.Q_corr[0].requires_grad:
            for i in range(self.ng):
                self.Q_corr[i] = nn.Parameter(self.Qfac/10*torch.randn(5*6//2, dtype=self.precision).to(self.device), requires_grad=True)
        if self.R_corr.requires_grad:
            self.R_corr = nn.Parameter(self.Rfac/10*torch.randn(3*4//2, dtype=self.precision).to(self.device), requires_grad=True)

        if self.ng>1 or not self.no_acc:
            for layers in self.layers:
                for l in layers:
                    l.reset_parameters()
        if self.extractor is not None:
            for l in self.extractor:
                if isinstance(l, nn.Linear):
                    l.reset_parameters()
        if self.head_layer is not None:
            self.head_layer.reset_parameters()
        # if self.Qnet is not None:
        #     self.Qnet.reset_parameters()
        # if self.Hnet is not None: # TODO requires Hnet.apply(reset_weights) to reset properly
        #     self.Hnet.reset_parameters()

    def initialize(self, target_id=None, **kwargs):
        Trackers.Tracker.initialize(self, target_id)
        self.init_state()
        self.eval()

    def init_state(self):
        self.t = 0
        self.base_g = 0
        self.x = (self.ng*[None]) if self.init_nan else [torch.zeros(6, dtype=self.precision).to(self.device) for _ in range(self.ng)]
        self.x_prev = max(1,self.input_obs-1)*[self.x[self.base_g]]
        self.P = [(self.Pfac**2)*torch.eye(6, dtype=self.precision).to(self.device) for _ in range(self.ng)]
        self.S = self.ng * [None]
        if self.dynamic_Q and (not self.frozen_Q_pred):
            self.Q = [(self.Qfac)*torch.eye(6 ,dtype=self.precision).to(self.device) for _ in range(self.ng)]
        if self.dynamic_R and (not self.const_R):
            self.R = self.Rfac * torch.eye(4 ,dtype=self.precision).to(self.device)
        self.p = torch.ones(self.ng, dtype=self.precision).to(self.device)/self.ng # [torch.tensor(1/self.ng).double() for _ in range(self.ng)]
        self.set_H()
        # last observation
        self.z = torch.zeros(4, dtype=self.precision).to(self.device)
        # hidden state variables: n_gaussians x acc (radial & theta & phi) & other hidden variables
        self.a = torch.zeros((self.ng,3), dtype=self.precision).to(self.device) # [torch.zeros(3, dtype=self.precision) for _ in range(self.ng)]
        self.h = [[torch.zeros_like(self.h[0][0], dtype=self.precision).to(self.device) for _ in range(len(self.h[0]))]
                  for _ in range(len(self.h))] # LSTM hidden state
        self.c = [[torch.zeros_like(self.c[0][j], dtype=self.precision).to(self.device) for j in range(len(self.c[0]))]
                  for _ in range(len(self.c))] # LSTM memory cell
        self.record = dict(g_pred=[],g_est=[],p_pred=[],p_est=[],x_pred=[],x_est=[],a=[],P_pred=[],P_est=[])

    def train(self, *args, **kwargs):
        nn.Module.train(self, *args, **kwargs)
        if self.record_history is None:
            self.recording = not self.training

    def eval(self):
        nn.Module.eval(self)
        self.Q = [self.get_cov_matrix('Q', i, force_calc=True).detach() for i in range(self.ng)]
        self.R = self.get_cov_matrix('R', force_calc=True).detach()
        if self.record_history is None:
            self.recording = not self.training

    def set_device(self, device=None):
        if device is None: device = self.device
        self.device = device
        if self.ng>1 or not self.no_acc:
            self.layers = nn.ModuleList([nn.ModuleList([l.to(self.device) for l in ls]) for ls in self.layers])
        self.init_state()

    def get_params_hash(self):
        return hex(hash(torch.cat([par.flatten() for par in self.parameters()]).detach().numpy().tostring()))

    def save_model(self, fname=None, base_path='data/models/'):
        if fname is None: fname = self.title
        if fname.endswith('.m'): fname = fname[:-2]
        # module state dict
        if list(self.parameters()):
            torch.save(self.state_dict(), base_path+fname+'.m')
        # noise estimation
        if self.const_R:
            torch.save((self.R_diag, self.R_corr), base_path+fname+'.R')
        if self.const_Q:
            torch.save((self.Q_diag, self.Q_corr), base_path+fname+'.Q')
        # features normalization
        if self.normalize_features:
            torch.save(self.normalize_features, base_path+fname+'.nrm')

    def load_model(self, fname=None, base_path='data/models/'):
        if fname is None: fname = self.title
        if fname.endswith('.m'): fname = fname[:-2]
        # module state dict
        NN = (self.ng>1) or (not self.no_acc) or (not self.const_Q) or (not self.const_R)
        if list(self.parameters()):
            self.load_state_dict(torch.load(base_path+fname+'.m'))
        # noise estimation
        if self.const_R:
            self.R_diag, self.R_corr = torch.load(base_path+fname+'.R')
            # R_diag, R_corr = torch.load(base_path+fname+'.R')
            # for i, (rd,rc) in enumerate(zip(R_diag,R_corr)):
            #     self.R_diag.append(copy.deepcopy(rd))
            #     self.R_corr.append(copy.deepcopy(rc))
        if self.const_Q:
            Q_diag, Q_corr = torch.load(base_path+fname+'.Q')
            self.Q_diag = [copy.deepcopy(q) for q in Q_diag]
            if not self.no_corr:
                self.Q_corr = [copy.deepcopy(q) for q in Q_corr]
        # if self.use_Q_estimation:
        #     Q_diag, Q_corr = torch.load(base_path+fname+'.Q')
        #     if self.const_Q:
        #         self.Q_diag = [copy.deepcopy(q) for q in Q_diag]
        #         if not self.no_corr:
        #             self.Q_corr = [copy.deepcopy(q) for q in Q_corr]
        #     else:
        #         for i in range(len(self.Q_diag)):
        #             self.Q_diag[i] = nn.Parameter(copy.deepcopy(Q_diag[i]))
        #         if not self.no_corr:
        #             self.Q_corr[i] = nn.Parameter(copy.deepcopy(Q_corr[i]))
        self.R = self.get_cov_matrix('R', force_calc=True).detach()
        self.Q = [self.get_cov_matrix('Q', i, force_calc=True).detach() for i in range(self.ng)]
        # mark Q,R as already-estimated to avoid overwriting
        if self.use_Q_estimation: self.estimated_Q = True
        if self.use_R_estimation: self.estimated_R = True
        # features normalization
        if self.normalize_features:
            self.normalize_features = torch.load(base_path+fname+'.nrm')
        # put (possibly-)loaded tensors on gpu
        if self.ng>1 or not self.no_acc:
            self.layers = nn.ModuleList([nn.ModuleList([l.to(self.device) for l in ls]) for ls in self.layers])

    def freeze_Q_pred(self):
        if not self.dynamic_Q:
            warn('freeze_Q_pred() is only intended for use under dynamic_Q.')
        self.frozen_Q_pred = True
        self.const_Q = True
        self.use_Q_estimation = True

    def unfreeze_Q_pred(self):
        if not self.dynamic_Q:
            warn('unfreeze_Q_pred() is only intended for use under dynamic_Q.')
        self.frozen_Q_pred = False
        self.const_Q = False
        self.use_Q_estimation = False

    def freeze_R_pred(self):
        if not self.dynamic_R:
            warn('freeze_R_pred() is only intended for use under dynamic_R.')
        self.const_R = True
        self.use_R_estimation = True

    def unfreeze_R_pred(self):
        if not self.dynamic_R:
            warn('unfreeze_R_pred() is only intended for use under dynamic_R.')
        self.const_R = False
        self.use_R_estimation = False

    def freeze_H_pred(self):
        if self.Hnet is None:
            warn('freeze_H_pred() is only intended for use with Hnet.')
        if self.pred_H > 0:
            self.pred_H = -self.pred_H

    def unfreeze_H_pred(self):
        if self.Hnet is None:
            warn('unfreeze_H_pred() is only intended for use with Hnet.')
        if self.pred_H <= 0:
            self.pred_H = -self.pred_H

    def forward(self, x):
        self.predict()
        self.do_update(x, trash_dim=len(x)>4)
        return self.get_pos()

    def predict(self, dt=None):
        if (not self.naive_tracker) and (self.x[self.base_g] is not None):
            self.hidden_predict(self.z)
            self.state_predict(dt=dt)
        self.t = self.t + 0.5

    def do_update(self, z, trash_dim=None):
        if trash_dim is None: trash_dim = len(z)>4
        if trash_dim:
            z = z[1:]
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z)
        if not torch.isnan(z[0]):
            if self.precision == torch.double:
                z = z.double()
            else:
                z = z.float()
            z = z.to(self.device)
            for i in range(self.update_iters):
                self.state_update(z)
        self.t = self.t + 0.5

    def get_rotation_matrix(self, i=None):
        # return R s.t. R*v = (|V|,0,0) and (R*x)[1] is horizontal.
        if i is None:
            i = self.base_g
        # v dir
        vhat = self.x[i][3:]
        norm = vhat.norm()
        if norm == 0:
            vhat = torch.tensor((1,0,0), dtype=self.precision)
        else:
            vhat = vhat / norm
        r1 = vhat
        # horizontal perp dir
        r2 = torch.tensor((r1[1],-r1[0],0), dtype=self.precision)
        r2 = r2 / r2.norm()
        # vertical perp dir
        r3 = torch.tensor((0,0,1), dtype=self.precision)
        r3 = r3 - r3.dot(r1)*r1
        r3 = r3 / r3.norm()

        return torch.stack((r1,r2,r3), dim=0)

    def get_features(self, z, i=None, x=None, eps=1e-5):
        # consider calculating features before the update-step / use characteristics of the last update.
        if x is None:
            x = self.x[i]
        dx = z[:3] - x[:3]
        dx_abs = dx.norm()
        vhat = x[3:]
        vnorm = vhat.norm()
        if vnorm != 0:
            vhat = vhat / vnorm
        dr = (dx*vhat).sum()
        dx_perp = (dx_abs**2-dr**2+eps).sqrt()
        if dx_abs!=0 and vnorm!=0:
            dtheta = ((torch.atan2(dx[1], dx[0]) - torch.atan2(vhat[1], vhat[0]) + np.pi) % (2 * np.pi)) - np.pi
            dphi = torch.acos(dx[2]/dx_abs) - torch.acos(vhat[2])
        else:
            dtheta = torch.tensor(0., dtype=self.precision).to(self.device)
            dphi = torch.tensor(0., dtype=self.precision).to(self.device)
        out = torch.stack((*dx, dx_abs, dr, dx_perp, dtheta, dphi), dim=0)
        return out
        # return torch.tensor([], dtype=self.precision).to(self.device)

    def set_features_normalizer(self, Z, X, verbose=0):
        X = torch.tensor(np.concatenate(X, axis=0))
        if X.shape[1] > 6:
            X = X[:,-6:]
        Z = torch.tensor(np.concatenate(Z, axis=0))
        if Z.shape[1] > 4:
            Z = Z[:,-4:]
        F = torch.stack([self.get_features(z,x=x) for z,x in zip(Z,X)], dim=0)
        if verbose >= 1:
            print('Obs/State/Features dimensions:', Z.shape, X.shape, F.shape)
        X = torch.cat((Z,*(self.input_obs*[X]),F), dim=1)
        if verbose >= 1:
            print('Total dimensions:', X.shape)
        self.normalize_features = [X.mean(dim=0).detach(), X.std(dim=0).detach()]
        if self.feed_t:
            self.normalize_features[0] = torch.cat((self.normalize_features[0], 30*torch.ones(1, dtype=self.precision, device=self.device)))
            self.normalize_features[1] = torch.cat((self.normalize_features[1], 30*torch.ones(1, dtype=self.precision, device=self.device)))
        if self.feed_P:
            # manually and roughly tuned for P.diag().log()
            self.normalize_features[0] = torch.cat((self.normalize_features[0], 5*torch.ones(6, dtype=self.precision, device=self.device)))
            self.normalize_features[1] = torch.cat((self.normalize_features[1], 1*torch.ones(6, dtype=self.precision, device=self.device)))
        if self.feed_Q:
            self.normalize_features[0] = torch.cat((self.normalize_features[0], 5*torch.ones(6, dtype=self.precision, device=self.device)))
            self.normalize_features[1] = torch.cat((self.normalize_features[1], 1*torch.ones(6, dtype=self.precision, device=self.device)))

    def get_network_input(self, z, i):
        R2 = R = None
        if self.rotate_input:
            R = self.get_rotation_matrix(i)
            ZERO = torch.zeros_like(R)
            R2 = torch.cat((torch.cat((R,ZERO),dim=1),torch.cat((ZERO,R),dim=1)))
        features = self.get_features(z, i)
        # t
        if self.feed_t:
            t = torch.tensor([self.t], dtype=self.precision, device=self.device)
        else:
            t = torch.tensor([], dtype=self.precision, device=self.device)
        # P
        if self.feed_P:
            # P = NeuralKF.encode_positive_definite(self.P[i])
            P = self.P[i]
            if self.rotate_input:
                P = R2.matmul(P).matmul(R2.T)
            P = P.diag()
            P = torch.clamp(P, 1, 1e8)
            P = P.log()
        else:
            P = torch.tensor([], dtype=self.precision, device=self.device)
        # Q
        if self.feed_Q:
            Q = self.Q[i]
            if self.rotate_input:
                Q = R2.matmul(Q).matmul(R2.T)
            Q = Q.diag().log()
        else:
            Q = torch.tensor([], dtype=self.precision, device=self.device)

        # concatenate
        x0 = torch.cat((z, self.x[i], *self.x_prev[:self.input_obs-1], t, P, Q, features)).view(1, -1)

        # normalize
        if self.normalize_features:
            x0 = (x0-self.normalize_features[0]) / self.normalize_features[1]

        # rotate
        if self.rotate_input:
            x0 = x0.view(-1)
            x0 = torch.cat((R.matmul(x0[:3]), x0[3:4], R2.matmul(x0[4:10]), x0[10:]))
            x0 = x0.view(1, -1)

        # extract input
        if self.extractor is not None:
            x0 = self.extractor(x0)
        return x0

    def hidden_predict(self, z, multi_input_states=None, separate_hidden_states=None,
                       aggregate_p=None, aggregate_P=False, manual_head_clip=None, eps=1e-3):
        if multi_input_states is None: multi_input_states = not self.merge_on_update
        if separate_hidden_states is None: separate_hidden_states = self.separate_hidden_states
        if aggregate_p is None: aggregate_p = self.aggregate_p
        if manual_head_clip is None: manual_head_clip = self.manual_head_clip

        # predict accelerations & probabilities
        if self.ng == 1 and self.no_acc:
            pass # a=0 and p=1 already

        else:
            x0 = None
            a = []
            p = []
            if not multi_input_states:
                x0 = self.get_network_input(z, self.base_g)
            for i in range(self.ng):
                # get prepapred
                if multi_input_states:
                    x0 = self.get_network_input(z, i)

                hid = i if separate_hidden_states else self.base_g

                # run LSTM
                for j in range(len(self.h[0])):
                    x = x0 if j==0 else self.h[i][j-1]
                    self.h[i][j], self.c[i][j] = self.layers[i][j](x, (self.h[hid][j], self.c[hid][j]))

                # run last LSTM layer
                x = self.h[i][-1]
                if self.pred_acc:
                    h = torch.cat((self.p[i].unsqueeze(0), self.a[hid,:]))
                    if self.dynamic_Q=='lstm':
                        if self.no_corr:
                            Q = (self.Q[i].diag()+eps)
                            if self.exp_cov_transform:
                                Q = Q.log()
                        else:
                            Q = self.encode_positive_definite(self.Q[i])
                        h = torch.cat((h, Q))
                    h = h.view(1,-1)
                else:
                    h = torch.cat((self.p[i].unsqueeze(0), self.x[hid]))
                    if self.no_corr:
                        P = (self.P[i].diag()+eps).sqrt()
                        if self.exp_cov_transform:
                            P = P.log()
                    else:
                        P = self.encode_positive_definite(self.P[i])
                    h = torch.cat((h, P)).view(1,-1)
                h, self.c[i][-1] = self.layers[i][-1](x, (h, self.c[hid][-1]))

                # run head layer
                d1 = 1
                if self.head_layer is None:
                    if manual_head_clip:
                        aa = 1e3*h[0, d1:]
                    else:
                        aa = [hh for hh in h[0, d1:]]
                        # numerically stable transform (-1,1) -> (-1/eps,1/eps)
                        for j in range(len(aa)):
                            aa[j] = aa[j] / (1 + eps - aa[j]) if aa[j] >= 0 else aa[j] / (1 + eps + aa[j])
                        aa = torch.stack(aa)
                else:
                    aa = self.head_layer(h[0, d1:])

                # extract probability
                p.append(h[0,0])
                if aggregate_p:
                    p[-1] = p[-1] + ((1-eps)*self.p[i]+eps).log().item()

                # extract acceleration
                # a0 = aa[0]/(1+eps-aa[0]) if aa[0]>= 0 else aa[0]/(1+eps+aa[0])
                # a1 = aa[1]/(1+eps-aa[1]) if aa[1]>= 0 else aa[1]/(1+eps+aa[1])
                # a2 = aa[2]/(1+eps-aa[2]) if aa[2]>= 0 else aa[2]/(1+eps+aa[2])
                if self.pred_acc:
                    a.append(torch.tensor(3*[0.],dtype=self.precision).to(self.device).unsqueeze(0) if self.no_acc else \
                                 aa[:3].unsqueeze(0))
                else:
                    self.x[i] = self.x[i] + aa[d1:d1+6]

                # extract process noise (Q)
                if self.pred_acc:
                    if self.dynamic_Q and (not self.frozen_Q_pred):
                        if self.dynamic_Q == 'lstm':
                            self.Q[i] = self.get_positive_definite(aa[3:3+(6 if self.no_corr else 21)], n=6)
                        elif self.dynamic_Q == 'nn':
                            self.Q[i] = self.Qnet(torch.cat((self.h[hid][-1], self.c[hid][-1]),dim=1)).exp()[0,:].diag()
                            if self.rotate_input:
                                R = self.get_rotation_matrix(hid)
                                ZERO = torch.zeros_like(R)
                                R2 = torch.cat((torch.cat((R,ZERO),dim=1),torch.cat((ZERO,R),dim=1)))
                                self.Q[i] = R2.T.matmul(self.Q[i]).matmul(R2)
                        else:
                            warn(f'invalid dynamic_Q value: {self.dynamic_Q}')
                else:
                    if aggregate_P:
                        raise NotImplementedError()
                    else:
                        self.P[i] = self.get_positive_definite(aa[6:6+21], n=6) # P*aa instead of aa?

            self.p = F.softmax(torch.stack(p), dim=0)
            if self.pred_acc:
                a = torch.cat(a, dim=0)
                if self.a_skip:
                    a0 = self.a if multi_input_states else torch.stack(self.ng*[self.a[self.base_g]])
                    self.a = a0 + a
                else:
                    self.a = a

        if self.recording:
            self.record['a'].append(self.a.detach().cpu().numpy())
            self.record['p_pred'].append(self.p.detach().cpu().numpy())

    def state_predict(self, dt=None, vectoric=True, eps=1e-6):
        if dt is None:
            dt = self.dt
            F = self.F
        else:
            F = torch.tensor([
                [1.,0,0,dt,0,0],
                [0,1,0,0,dt,0],
                [0,0,1,0,0,dt],
                [0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
            ], dtype=self.precision).to(self.device)

        if self.pred_acc:
            # get current gaussian
            x = self.x[self.base_g]
            vrel = x[3:]
            for i in range(3):
                # for numeric stability
                if 0 <= vrel[i] < eps:
                    vrel[i] = eps
                elif -eps < vrel[i] < 0:
                    vrel[i] = -eps
            vnorm = vrel.norm()
            if vnorm != 0:
                vrel = vrel / vnorm
            vrelxy = vrel[:2]
            vnorm = vrelxy.norm()
            if vnorm != 0:
                vrelxy = vrelxy / vnorm

            if vectoric:
                # get accelerations
                if self.radial_acc:
                    a1 = self.a[:,0] * vrel[0] - self.a[:,1] * vrel[1] - self.a[:,2] * vrelxy[0] * vrel[2]
                    a2 = self.a[:,0] * vrel[1] + self.a[:,1] * vrel[0] - self.a[:,2] * vrelxy[1] * vrel[2]
                    a3 = self.a[:,0] * vrel[2] + self.a[:,2] * (1 - vrel[2] ** 2).sqrt()
                else:
                    a1 = self.a[:,0]
                    a2 = self.a[:,1]
                    a3 = self.a[:,2]
                # predict x
                dt2 = dt ** 2
                for i in range(self.ng):
                    self.x[i] = torch.stack([
                        x[0] + x[3 + 0] * dt + 0.5 * a1[i] * (dt2),
                        x[1] + x[3 + 1] * dt + 0.5 * a2[i] * (dt2),
                        x[2] + x[3 + 2] * dt + 0.5 * a3[i] * (dt2),
                        x[3] + a1[i] * dt,
                        x[4] + a2[i] * dt,
                        x[5] + a3[i] * dt
                    ])

            else:
                for i in range(self.ng):
                    # get accelerations
                    a1 = self.a[i][0]*vrel[0] - self.a[i][1]*vrel[1] - self.a[i][2]*vrelxy[0]*vrel[2]
                    a2 = self.a[i][0]*vrel[1] + self.a[i][1]*vrel[0] - self.a[i][2]*vrelxy[1]*vrel[2]
                    a3 = self.a[i][0]*vrel[2] + self.a[i][2]*(1-vrel[2]**2).sqrt()
                    # predict x
                    dt2 = dt**2
                    self.x[i] = torch.stack([
                        x[0] + x[3+0]*dt + 0.5*a1*(dt2),
                        x[1] + x[3+1]*dt + 0.5*a2*(dt2),
                        x[2] + x[3+2]*dt + 0.5*a3*(dt2),
                        x[3] + a1*dt,
                        x[4] + a2*dt,
                        x[5] + a3*dt
                    ])
            # predict P
            for i in range(self.ng):
                Q = self.get_cov_matrix(case='Q', i=i)
                self.P[i] = mp(mp(F, self.P[i]), F.T) + Q
                self.P[i] = 0.5*(self.P[i]+self.P[i].T) # numeric stability: force symmetric P

        self.base_g = self.p.argmax().item()

        if self.recording:
            self.record['x_pred'].append([x.detach().cpu().numpy() for x in self.x])
            self.record['P_pred'].append([P.detach().cpu().numpy() for P in self.P])
            self.record['g_pred'].append(self.base_g)

    def state_update(self, z, H_from_obs=None, update_all_gaussians=None, eps=1e-6):
        if H_from_obs is None: H_from_obs = self.H_from_obs
        if update_all_gaussians is None:
            update_all_gaussians = (not self.merge_on_update) or self.training
        self.z = z

        if self.naive_tracker:
            self.x[0][:3] = z[:3]
        else:
            # choose x (most likely gaussian)
            nll = self.calc_nlls(z)
            self.base_g = torch.argmin(nll).item()
            self.p = F.softmax(-nll, dim=0)

            gaussians = range(self.ng) if update_all_gaussians else [self.base_g]

            for i in gaussians:
                self.set_H(pos=z[:3] if H_from_obs else None)

                if self.dynamic_R and (not self.const_R) and self.x[self.base_g] is not None:
                    inp = self.get_input_for_update()
                    R = self.Rnet(inp).exp()[0,:].diag()
                    if self.rotate_input:
                        rot = self.get_rotation_matrix(i)
                        ZERO = torch.tensor([0,0,0,1], device=self.device).view(-1,1)
                        if self.precision == torch.double: ZERO = ZERO.double()
                        rot = torch.cat((torch.cat((rot,ZERO[:-1]),dim=1), ZERO.T))
                        R = rot.T.matmul(R).matmul(rot)
                else:
                    R = self.get_cov_matrix(case='R')

                if self.polar_R:
                    J = self.get_J_p2c(z[0], z[1], z[2])
                    R = mp(mp(J, R), J.T)
                # get update operators
                Ht = self.H.T
                PHt = mp(self.P[i], Ht)
                self.S[i] = mp(self.H, PHt) + R
                K = mp( PHt, self.S[i].inverse() )
                # update P
                I_KH = torch.eye(self.P[i].shape[0]).to(self.device) - mp(K, self.H)
                # I_KH = torch.eye(self.P[i].shape[0]) - mp(K, self.H)
                # self.P[i] = mp(torch.eye(self.P[i].shape[0])-mp(K,self.H), self.P[i])
                self.P[i] = mp(mp(I_KH,self.P[i]),I_KH.T) + mp(mp(K,R),K.T) # numeric stability: equivalent but more stable formula
                self.P[i] = 0.5*(self.P[i]+self.P[i].T) # numeric stability: force symmetric P
                # update x
                if self.x[i] is None:
                    x = z[:3]
                    v = x / (x.norm()+eps) * z[-1]
                    self.x[i] = torch.cat((x,v))
                    # self.P[i] = mp(mp(Ht,R),self.H) # this is wrong initialization because it cannot express the uncertainty in speed directions orthogonal to doppler
                else:
                    self.x[i] = self.x[i] + mp(K, z-mp(self.H,self.x[i]))

        x_prev = self.x[self.base_g] # (it will become "prev" in retrospect after next predict...)
        self.x_prev = [x_prev] + self.x_prev[:-1]

        if self.recording:
            self.record['x_est'].append([x if x is None else x.detach().cpu().numpy() for x in self.x])
            self.record['P_est'].append([P if P is None else P.detach().cpu().numpy() for P in self.P])
            self.record['p_est'].append(self.p.detach().cpu().numpy())
            self.record['g_est'].append(self.base_g)

    def calc_S(self, gaussians=None):
        if gaussians is None:
            gaussians = range(self.ng)
        if self.H is None:
            self.set_H()
        for i in gaussians:
            PHt = mp(self.P[i], self.H.T)
            R = self.get_cov_matrix(case='R')
            if self.polar_R:
                if self.x[self.base_g] is None:
                    J = self.get_J_p2c(0,0,0)
                else:
                    J = self.get_J_p2c(self.x[self.base_g][0], self.x[self.base_g][1], self.x[self.base_g][2])
                R = mp(mp(J, R), J.T)
            self.S[i] = mp(self.H, PHt) + R

    def calc_nlls(self, z, prior=True, include_doppler=None, obs=None, gaussians=None, eps=1e-6):
        if gaussians is None:
            gaussians = range(self.ng)
        if include_doppler is None:
            include_doppler = self.full_nll_train if self.training else self.full_nll_eval
            if obs is None:
                obs = (len(z) < 6)
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=self.precision)
        if self.precision == torch.double:
            z = z.double()
        else:
            z = z.float()
        z = z.to(self.device)
        nll = []

        if self.x[self.base_g] is None:
            return torch.zeros(self.ng, dtype=self.precision, device=self.device)

        if self.naive_tracker:
            nll.append(((z[:3]-self.x[0][:3])**2).sum().unsqueeze(0))
            return torch.stack(nll)

        if include_doppler:
            if obs:
                # likelihood of an observation (4D)
                d = 4
                if len(z) == 3:
                    # need to "generate" doppler observation
                    warn('nll(z) should receive a 4D observation, not 3D. using estimated doppler entry instead.')
                    v = (z - self.x_prev[0][:3].detach())
                    v = (v*z).sum()/(z.norm()+eps) # projection of motion v on observation direction z
                    z = torch.cat((z, v.unsqueeze(0)))
                elif len(z) == 6:
                    # need to convert a 6D state x to 4D observation z
                    v = z[3:]
                    v = (v*z[:3]).sum()/(z[:3].norm()+eps)  # projection of motion v on observation direction z
                    z = torch.cat((z[:3], v.unsqueeze(0)))
                for i in gaussians:
                    if self.S[i] is None: self.calc_S([i])
                    A = -torch.log((2 * np.pi) ** (-d / 2) * self.S[i].det() ** (-0.5))
                    x = mp(self.H, self.x[i])
                    tmp = A + 0.5*mp( mp((z-x),self.S[i].inverse()), z-x)
                    if prior: tmp = tmp - self.p[i].log()
                    nll.append(tmp)

            else:
                # likelihood of a state (6D)
                d = 6
                for i in gaussians:
                    A = -torch.log((2 * np.pi) ** (-d / 2) * (self.P[i]+eps*torch.eye(d)).det() ** (-0.5))
                    tmp = A + 0.5 * mp(mp((z-self.x[i]), (self.P[i]+eps*torch.eye(d)).inverse()), z-self.x[i])
                    if prior: tmp = tmp - self.p[i].log()
                    nll.append(tmp)

        else:
            # likelihood of a location (3D) (either observed or true - no difference...)
            d = 3
            for i in gaussians:
                A = -torch.log((2 * np.pi) ** (-d / 2) * (self.P[i][:d,:d]+eps*torch.eye(d)).det() ** (-0.5))
                tmp = A + 0.5*mp( mp((z[:d]-self.x[i][:d]),(self.P[i][:d,:d]+eps*torch.eye(d)).inverse()), z[:d]-self.x[i][:d])
                if prior: tmp = tmp - self.p[i].log()
                nll.append(tmp)

        return torch.stack(nll)

    def calc_nll(self, z, prior=True, approx=False):
        nlls = self.calc_nlls(z, prior=prior)
        if approx:
            # simply use the most likely gaussian (good approximation if the gaussians are far from each other)
            return nlls.min().item()

        # min_nll is a numeric stabilizer: https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        min_nll = nlls.min()
        out = min_nll - torch.log(torch.sum(torch.exp(-(nlls - min_nll))))
        if torch.isinf(out):
            # nll>>1  =>  exponent(-nll)=0  =>  just take nll of most likely gaussian
            return nlls.min().item()
        return out.item()

    def get_pos(self, ML=None):
        if self.x[self.base_g] is None: return np.zeros(3)
        if ML is None: ML = self.ML_estimator
        if ML:
            return self.x[self.base_g][:3].detach().cpu().numpy()
        else:
            return torch.stack([self.p[i]*self.x[i][:3] for i in range(self.ng)]).sum(dim=0).detach().cpu().numpy()

    def get_cov_matrix(self, case='Q', i=None, force_calc=False):
        if i is None:
            i = self.base_g

        if not force_calc:
            if (not self.training) or (self.dynamic_Q and (not self.frozen_Q_pred)):
                if case=='Q' and self.Q[i] is not None:
                    return self.Q[i]
                if case=='R' and self.R is not None:
                    return self.R

        if case=='Q':
            ii = i if self.multi_Q else 0
            Q_diag, Q_corr = self.Q_diag[ii], self.Q_corr[ii]
        elif case=='R':
            Q_diag, Q_corr = self.R_diag, self.R_corr
        else:
            raise ValueError(case)
        return self.get_positive_definite(torch.cat((Q_diag,Q_corr)), exp_diag=True, n=Q_diag.shape[0])

    def get_positive_definite(self, x, exp_diag=True, n=None): # more correct
        if n is None:
            n = int(np.round(-0.5+np.sqrt(0.25+2*len(x))))
        # positive entries on diagonal
        D = x[:n]
        if exp_diag and self.exp_cov_transform:
            D = D.exp() #if self.exp_cov_transform else torch.nn.functional.softplus(D)
        L = D.diag()
        # fill in lower triangular matrix
        if len(x) > n:
            ids = torch.tril_indices(n,n,-1)
            L[ids[0,:],ids[1,:]] = x[n:]
            # LL^T is positive definite
            A = mp(L, L.T)
        else:
            A = L
        return A

    def encode_positive_definite(self, A, exp_diag=True, my_cholesky=False, eps=1e-6):
        n = A.shape[0]
        if my_cholesky:
            # more control of nans
            L = self.cholesky(A+eps*torch.eye(n))
        else:
            L = torch.cholesky(A+eps*torch.eye(n))
        D = L.diag()
        if exp_diag:
            if self.exp_cov_transform:
                D = D.log()
        ids = torch.tril_indices(n,n,-1)
        L = L[ids[0,:],ids[1,:]]
        return torch.cat((D,L))

    # cholesky is implemented manually instead of torch.cholesky() to avoid errors on bad conditional numbers
    # source: https://stackoverflow.com/questions/60230464/pytorch-torch-cholesky-ignoring-exception
    def cholesky(self, A):
        n = A.shape[1]
        L = [[torch.tensor(0, dtype=self.precision).to(self.device) for _ in range(n)] for _ in range(n)]
        for i in range(A.shape[1]):
            for j in range(i+1):
                s = 0.
                for k in range(j):
                    s = s + L[i][k] * L[j][k]
                L[i][j] = torch.sqrt(A[i,i]-s) if i==j else (1/L[j][j]*(A[i,j]-s))
        L = torch.stack([torch.stack([torch.tensor(0, dtype=self.precision).to(self.device) if torch.isnan(l) or torch.isinf(l) else l
                                      for l in row], dim=0) for row in L], dim=0)
        return L

    def get_input_for_update(self, i=None):
        if i is None: i = self.base_g
        x = torch.cat((self.x[i][:3]/1e3, self.x[i][3:]/1e2))
        z = torch.cat((self.z[:3]/1e3, self.z[3:]/1e2))
        if self.rotate_input:
            R = self.get_rotation_matrix()
            ZERO = torch.zeros_like(R)
            R2 = torch.cat((torch.cat((R,ZERO),dim=1),torch.cat((ZERO,R),dim=1)))
            x = torch.cat((R.matmul(x[:3]), R.matmul(x[3:])))
            z = torch.cat((R.matmul(z[:3]), z[3:]))
            P = R2.matmul(self.P[i]).matmul(R2.T)
        else:
            P = self.P[i]
        P = P.diag().log() - 5

        x = x.view(1, -1)
        z = z.view(1, -1)
        P = P.view(1, -1)

        return torch.cat((x,z,P), dim=1)

    def set_H(self, pos=None):
        self.H = self.get_H(pos)

    def get_H(self, pos=None):
        if self.pred_H>0 and self.x[self.base_g] is not None:
            H = torch.tensor([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ], dtype=self.precision).to(self.device)

            inp = self.get_input_for_update()

            out = self.Hnet(inp)
            if self.rotate_input:
                R = self.get_rotation_matrix()
                if out.numel() == 3:
                    out = R.T.matmul(out.view(-1))
                else:
                    out = torch.cat((R.T.matmul(out[0,:3].view(-1)), R.T.matmul(out[0,3:].view(-1))))
            if out.numel() == 3:
                H[-1,3:] = out
            else:
                H[-1,:] = out
            return H

        if self.EKF:
            # EKF derivation
            state = torch.zeros(6, dtype=self.precision, device=self.device) \
                if self.x[self.base_g] is None else self.x[self.base_g]
            x, y, z = state[:3]
            vx, vy, vz = state[3:]

            r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
            p = x*vx+y*vy+z*vz
            if r == 0:
                return torch.tensor([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0]
                ], dtype=self.precision).to(self.device)
            else:
                return torch.tensor([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [vx/r-p*x/r**3, vy/r-p*y/r**3, vz/r-p*z/r**3, x/r, y/r, z/r]
                ], dtype=self.precision).to(self.device)

        else:
            if pos is None:
                if self.x[self.base_g] is None:
                    pos = torch.tensor((0,0,0), dtype=self.precision, device=self.device)
                else:
                    pos = self.x[self.base_g][:3]
            x,y,z = pos
            r = torch.sqrt(x**2+y**2+z**2)
            if r == 0:
                return torch.tensor([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ], dtype=self.precision).to(self.device)
            else:
                return torch.tensor([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, x/r, y/r, z/r]
                ], dtype=self.precision).to(self.device)

    def get_J_p2c(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        xy = np.sqrt(x**2 + y**2)
        ct = x / xy
        st = y / xy
        cp = z / r if r!=0 else 1
        sp = xy / r if r!=0 else 0
        J = np.array([
            [sp*ct, -r*sp*st, r*cp*ct, 0],
            [sp*st,  r*sp*ct, r*cp*st, 0],
            [cp,     0,       -r*sp,   0],
            [0,      0,       0,       1]
        ])
        return torch.tensor(J)

    def get_SE(self, y):
        return np.sum((self.get_pos()-y)**2)

    def loss_pred(self, y, detailed=False, approx=False, force=False):
        y = torch.tensor(y, dtype=self.precision, device=self.device)
        lamda = 1-self.update_loss_fac
        if (lamda == 0) and not force:
            out = torch.tensor(0, dtype=self.precision, device=self.device)
            g = self.base_g
        else:
            nlls = self.calc_nlls(y)
            g = nlls.argmin().item()
            # min_nll is a numeric stabilizer: https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
            min_nll = nlls.min()
            out = min_nll - torch.log(torch.sum(torch.exp(-(nlls-min_nll))))
            if approx or torch.isinf(out):
                # nll>>1  =>  exponent(-nll)=0  =>  just take nll of most likely gaussian
                out = torch.min(nlls)
        outn = lamda * out
        if detailed:
            return (outn, out, g)
        return outn, out

    def loss_update(self, y, detailed=False, weighted=False, p=None, force=False):
        if p is None: p = self.update_loss_p
        y = torch.tensor(y[:3], dtype=self.precision, device=self.device)
        lamda = self.update_loss_fac
        if (lamda == 0) and not force:
            out = torch.tensor(0, dtype=self.precision, device=self.device)
        else:
            if weighted:
                out = torch.stack([prob*((x[:3]-y)**2).sum() for x,prob in zip(self.x,self.p)]).sum()
            else:
                out = ((self.x[self.base_g][:3]-y)**2).sum()
        outn = lamda * (out**(p/2))
        if detailed:
            return (outn, out, self.base_g)
        return outn, out

    def estimate_R(self, Z, X, inplace=True, force=False, polar=None, H_from_z=None):
        '''
        Estimate the observation noise covariance matrix R from data.
        :param Z: observations data (Nx4)
        :param X: state data (Nx6)
        '''
        if H_from_z is None: H_from_z = self.estimate_R_from_HZ
        if polar is None: polar = self.polar_R
        if not isinstance(Z[0], np.ndarray):
            Z = [np.array(z) for z in Z]
        if not isinstance(X[0], torch.Tensor):
            X = [torch.tensor(x) for x in X]
        Z = np.concatenate(Z, axis=0)
        X = torch.cat(X, dim=0)
        if X.shape[1] > 6:
            X = X[:, -6:]
        if Z.shape[1] > 4:
            Z = Z[:, -4:]

        ids = np.all(np.logical_not(np.isnan(Z)), axis=1)
        Z = Z[ids, :]
        X = X[ids, :]

        H = [self.get_H(p[:3]) for p in (torch.tensor(Z) if H_from_z else X)]
        Xz = np.stack([mp(h,x).detach().numpy() for x,h in zip(X,H)], axis=0)
        if polar:
            Z[:,:3] = cart2polar(Z[:,:3])
            Xz[:,:3] = cart2polar(Xz[:,:3])

        delta = Z - Xz
        if polar:
            ids_neg = delta[:,1] < -np.pi
            ids_pos = delta[:,1] > np.pi
            delta[ids_neg, 1] = delta[ids_neg, 1] + 2*np.pi
            delta[ids_pos, 1] = delta[ids_pos, 1] - 2*np.pi

        R = torch.tensor(np.cov((delta).T), dtype=self.precision)

        if self.cheat_R:
            R[:] = 0
            R[0,0] = 10
            R[1,1] = 1/180*3.14
            R[2,2] = 3/180*3.14
            R[3,3] = 5
            R = R**2

        if inplace and (force or not self.estimated_R):
            self.R = R
            R = self.encode_positive_definite(R)
            if self.const_R:
                self.R_diag = R[:4]
                if not self.no_corr:
                    self.R_corr = R[4:]
            else:
                with torch.no_grad():
                    self.R_diag.copy_(R[:4])
                    if not self.no_corr:
                        self.R_corr.copy_(R[4:])
            self.estimated_R = True
            # self.R_diag = R.diag().sqrt().log().to(self.device)
            # n = self.R_diag.shape[0]
            # S = R.diag().sqrt().diag().inverse()
            # R = mp(mp(S, R), S) # cov to cor
            # ids = torch.tril_indices(n, n, -1)
            # R = R[ids[0,:],ids[1,:]] # under the diag
            # R = 0.5*((1+R)/(1-R)).log() # atanh (from pytorch 1.6 could use torch.atanh())
            # self.R_corr = R
        else:
            return R

    def estimate_Q(self, X, inplace=True, force=False):
        '''
        Estimate the observation noise covariance matrix R from data.
        :param X: state data (Nx6)
        '''
        if not isinstance(X[0], torch.Tensor):
            X = [torch.tensor(x) for x in X]
        X1 = torch.cat([x[:-1] for x in X], dim=0) # t
        X2 = torch.cat([x[1:]  for x in X], dim=0) # t+1

        if X1.shape[1]==6:
            X1 = mp(self.F, X1.T).T # prediction t+1|t
        elif X1.shape[1]==7:
            # apply F(dt(i)) for each X_i separately: pos_i += dt(i) * v_i
            dt_arr = torch.cat([x[1:,0]-x[:-1,0] for x in X])
            X1 = X1[:,1:]
            X2 = X2[:,1:]

            dt_arr = dt_arr.repeat(3,1).T
            X1[:,:3] += dt_arr * X1[:,3:6]
        else:
            raise ValueError(X1.shape)

        Q = torch.tensor(np.cov((X1-X2).T.detach().cpu().numpy()), dtype=self.precision)

        if inplace and (force or not self.estimated_Q):
            for i in range(self.ng):
                self.Q[i] = Q
            Q = self.encode_positive_definite(Q)
            if self.const_Q:
                for i in range(self.ng):
                    self.Q_diag[i] = Q[:6]
                    if not self.no_corr:
                        self.Q_corr[i] = Q[6:]
            else:
                with torch.no_grad():
                    for i in range(self.ng):
                        self.Q_diag[i].copy_(Q[:6])
                        if not self.no_corr:
                            self.Q_corr[i].copy_(Q[6:])
            self.estimated_Q = True

        else:
            return Q

def cart2polar(X):
    # X: nx3 (x,y,z)
    r = np.linalg.norm(X,axis=1)
    th = np.arctan2(X[:,1], X[:,0])
    phi = np.arccos(X[:,2]/r)
    if np.any(np.isnan(phi)):
        phi = np.nan_to_num(phi, nan=0, posinf=0, neginf=0)
    return np.stack((r,th,phi)).T

def polar2cart(X):
    # X: nx3 (r,theta,phi)
    x = X[:,0] * np.sin(X[:,2]) * np.cos(X[:,1])
    y = X[:,0] * np.sin(X[:,2]) * np.sin(X[:,1])
    z = X[:,0] * np.cos(X[:,2])
    return np.stack((x,y,z)).T
