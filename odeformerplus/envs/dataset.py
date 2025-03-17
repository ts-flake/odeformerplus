from __future__ import annotations
import json
import math
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from .symbolic import SymbolicExpression, SEP_TOKEN
from .generators import ExpressionTree, _traverse
from .utils import get_grid, print_dataset_stats

###########################
## DATASET VISUALIZATION ##
###########################

def plot_trajs(time, trajs, ax=None, figsize=(3,3), max_num=10, cmap='tab10', **kwargs):
    if ax is None: fig,ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max_num))
    
    for i,traj in enumerate(trajs.T):
        ax.plot(time, traj, color=colors[i], **kwargs)
    ax.set(xlabel='time', ylabel='x')
    try: return fig,ax
    except: return ax

def plot_dataset(ds, ncols=3, figsize=None, idx=None, **kwargs):
    idx = np.random.choice(len(ds)) if idx is None else idx
    print('index:',idx)
    data = ds[idx]

    SymbolicExpression().print_readable_sequence(data['ode'])

    trajs = data['trajectories']
    fig,axs = get_grid(len(trajs), ncols=ncols, figsize=figsize, return_fig=True)

    for traj,ax in zip(trajs, axs):
        traj = np.array(traj)
        time = traj[:,0]
        traj = traj[:,1:]
        plot_trajs(time, traj, ax=ax, **kwargs)

    fig.tight_layout(pad=0.9)
    return fig

def plot_2d_line_field_w_traj(data, figsize=(3,3), line_lw=1, traj_lw=2, xlim=None, ylim=None, axis_equal=True):
    fig,ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
    
    SymbolicExpression().print_readable_sequence(data['ode'])
    
    lines = data['lines']
    trajs = data['trajectories']
    
    # lines
    for line in lines:
        line = np.array(line)
        ax.plot(line[:,1], line[:,2], color='grey', lw=line_lw, alpha=0.8)
    
    # trajectories
    for i,traj in enumerate(trajs):
        traj = np.array(traj)
        ax.plot(traj[:,1], traj[:,2], lw=traj_lw, color=colors[i], alpha=1)
        ax.plot(traj[0,1], traj[0,2], '*', color=colors[i], alpha=1)
    
    ax.set(xlabel='x1', ylabel='x2')
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.set(ylim=ylim)
    if axis_equal:
        ax.set_aspect('equal')
    # fig.tight_layout(pad=0.9)
    return fig


#######################################
## DATASET, BATCHSAMPLER, COLLATE_FN ##
#######################################

class SymbolicRegressionDataset(Dataset):
    def __init__(self, file_name, stats_file_name=None):
        with open(file_name, 'r') as f:
            self.data = json.load(f)
            print('Loaded dataset from', file_name)
        
        if not stats_file_name:
            stats_file_name = Path(file_name).parent / 'stats_all.json'
        try:    
            with open(stats_file_name, 'r') as f:
                self.stats = json.load(f)
                print('Loaded stats from', stats_file_name)
        except:
            print('No stats found')
            self.stats = None

        
        # for plot 2d line field with trajectories
        self.idx2 = [idx for idx, data in enumerate(self.data) if data['dim'] == 2]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        if type(idx) == list:
            return [self.data[i] for i in idx]
        return self.data[idx]
    
    # TODO: add print stats and plot
    def print_stats(self):
        if self.stats is None:
            print('No stats found')
        else:
            print_dataset_stats(self.stats)

    def plot_dataset(self, ncols=3, figsize=None, idx=None, **kwargs):
        return plot_dataset(self, ncols, figsize, idx, **kwargs)
    
    def plot_2d_line_field_w_traj(self, figsize=(3,3), line_lw=1, traj_lw=2, xlim=None, ylim=None, axis_equal=True):
        if len(self.idx2) == 0:
            print('No 2d data found')
            return
        idx = np.random.choice(self.idx2)
        print('index:', idx)
        return plot_2d_line_field_w_traj(self[idx], figsize, line_lw, traj_lw, xlim, ylim, axis_equal)

class BucketBatchSampler(Sampler):
    "Batch sampler that groups input samples with similar lengths."
    def __init__(self, data_source, batch_size, bucket_width, drop_last=False, shuffle=True):
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.drop_last = drop_last
        self.shuffle = shuffle

        self._all_lengths = [self._count_length(data) for data in data_source]

        self.buckets = self._create_buckets() # if bucket_width=50, buckets = {0: [...], 50: [...], ...}
        self._buckets_key = list(self.buckets.keys())

    def _create_buckets(self):
        buckets = defaultdict(list)
        _sorted_idxs = np.argsort(self._all_lengths)
        for idx in _sorted_idxs:
            length = self._all_lengths[idx]
            bucket = (length // self.bucket_width) * self.bucket_width
            buckets[bucket].append(idx)
        return {k:v for k,v in sorted(buckets.items())} # sort buckets by key

    def _count_length(self, data):
        return sum(len(traj) for traj in data['trajectories'])
    
    def __iter__(self):
        if self.shuffle: np.random.shuffle(self._buckets_key)
        for key in self._buckets_key:
            idxs = self.buckets[key]
            if self.shuffle: np.random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch_idxs = idxs[i:i+self.batch_size]
                if len(batch_idxs) == self.batch_size or not self.drop_last:
                    yield batch_idxs
    
    def __len__(self):
        num_batches = 0
        for bucket in self.buckets:
            if self.drop_last:
                num_batches += len(self.buckets[bucket]) // self.batch_size
            else:
                num_batches += math.ceil(len(self.buckets[bucket]) / self.batch_size)
        return num_batches

def collate_fn_trajs(batch):
    trajs = []
    odes = []
    for data in batch:
        trajs.append([np.array(traj) for traj in data['trajectories']])
        odes.append(data['ode'])

    return{
        'trajectories': trajs,
        'odes': odes
    }

def collate_fn_lines(batch):
    lines = []
    odes = []
    for data in batch:
        lines.append(np.array(data['lines']))
        odes.append(data['ode'])

    return{
        'lines': lines,
        'odes': odes
    }

def collate_fn_trajs_lines(batch):
    trajs = []
    lines = []
    odes = []
    for data in batch:
        trajs.append([np.array(traj) for traj in data['trajectories']])
        lines.append(np.array(data['lines']))
        odes.append(data['ode'])

    return{
        'trajectories': trajs,
        'lines': lines,
        'odes': odes
    }

def add_noise_to_trajs(trajs: List[np.ndarray], noise_sig=0.1, deterministic=False):
    "Add multiplicative Gaussian noise to each trajectory."
    assert noise_sig > 0
    _trajs = []
    noise_sig = np.random.uniform(0, noise_sig) if not deterministic else noise_sig
    for traj in trajs:
        noise = np.random.normal(0, noise_sig, size=(traj.shape[0], traj.shape[1]-1))
        
        # add noise
        # traj: (time, x1, x2, ...)
        traj[:,1:] = (1 + noise) * traj[:,1:]
        _trajs.append(traj)
    return _trajs

def subsample_trajs(trajs: List[np.ndarray], drop_rate=0.5, deterministic=False):
    "Subsample trajectories."
    assert 0 < drop_rate < 1
    _trajs = []
    drop_rate = np.random.uniform(0, drop_rate) if not deterministic else drop_rate # percentage to drop
    for traj in trajs:
        num_drop = int(drop_rate * len(traj))
        msk = np.ones(len(traj), dtype=bool)
        msk[np.random.choice(len(msk), num_drop, replace=False)] = False
        
        # subsample
        _trajs.append(traj[msk,:])
    return _trajs

def rescale_trajs(trajs: List[np.ndarray], x0_radius: float, t_range: List[float]):
    "Rescale trajectories to the given range."
    # find the min and max of time and x
    _trajs = []
    t_min, t_max = 1e4, -1e4
    x0_max = 0
    for traj in trajs:
        t_min = min(t_min, traj[0,0])
        t_max = max(t_max, traj[-1,0])
        x0_max = max(x0_max, np.linalg.norm(traj[0,1:]))

    for traj in trajs:
        # rescale time
        t_norm = (traj[:,0] - t_min) / (t_max - t_min)
        _time = t_norm * (t_range[1] - t_range[0]) + t_range[0]

        # rescale x
        x_norm = traj[:,1:] / x0_max
        _x = x_norm * x0_radius
        _trajs.append(np.concatenate([_time[:,None], _x], axis=1))
    
    # affine transformation that restores the original range
    # t_a * t_rescaled + t_b = t_original
    # x_a * x_rescaled + x_b = x_original
    info = {
        't_a': (t_max - t_min) / (t_range[1] - t_range[0]),
        't_b': t_min - t_range[0] * (t_max - t_min) / (t_range[1] - t_range[0]),
        'x_a': x0_max / x0_radius,
        'x_b': 0
    }
    return _trajs, info
        
def subsample_exprs(exprs: List[str], drop_rate=0.2, mask_rate=0.2):
    "Subsample expressions."
    def is_number(s: str):
        try: float(s); return True
        except: return False

    _exprs = []
    expr = []
    for token in exprs + [SEP_TOKEN]:
        if token == SEP_TOKEN:
            # get tree
            tree = ExpressionTree.prefix_sequence_to_tree(expr)
            nodes = _traverse(tree, [], (lambda node: not node.is_root and not node.is_leaf))
            if not nodes:
                _exprs.extend(['<mask>', SEP_TOKEN])
                expr = []
            else:
                # randomly mask expression
                if np.random.rand() < mask_rate:
                    _exprs.extend(['<mask>', SEP_TOKEN])
                    expr = []
                # randomly drop nodes
                else:
                    for node in nodes:
                        if np.random.rand() < drop_rate:
                            if node.parent.lchild == node:
                                node.parent.lchild = None
                            else:
                                node.parent.rchild = None
                
                    # extract subtree with probability inversely proportional to its depth
                    depths = np.array([ExpressionTree.depth(node) for node in nodes])
                    probs = 1 / depths
                    node = np.random.choice(nodes, p=probs / sum(probs))
                    _exprs.extend(node.tree_to_prefix_sequence() + [SEP_TOKEN])
                    expr = []
        else:
            # replace number with '<const>'
            if is_number(token):
                token = '<const>'
            expr.append(token)
    return _exprs[:-1] # remove the last SEP_TOKEN
        
                
