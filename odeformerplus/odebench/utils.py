import re
import sympy as sy
import numpy as np
from scipy.integrate import solve_ivp
from copy import deepcopy
from odeformerplus.envs.symbolic import SymbolicExpression
from odeformerplus.envs.generators import DatasetGenerator
from odeformerplus.envs.dataset import add_noise_to_trajs, subsample_trajs, rescale_trajs, plot_trajs

from odeformerplus.envs.utils import load_config, get_grid
import matplotlib.pyplot as plt
config = load_config('odeformerplus/envs/config.yaml')

TIME_RANGR_MIN = config.dataset_generator.traj_time_range_l
TIME_RANGR_MAX = config.dataset_generator.traj_time_range_u
X0_RADIUS = config.dataset_generator.x0_range_traj

from .strogatz_equations import equations as ODEBENCH
OPTIONS = {
    "t_span": (0, 10),  # time span for integration
    "method": "LSODA",  # method for integration
    "rtol": 1e-5,  # relative tolerance (let's be strict)
    "atol": 1e-7,  # absolute tolerance (let's be strict)
    "first_step": 1e-6,  # initial step size (let's be strict)
    "t_eval": np.linspace(0, 10, 150),  # output times for the solution
    "min_step": 1e-10,  # minimum step size (only for LSODA)
}

def solve_odebench(idx: int, noise_sig: float=0.05, drop_rate: float=0.2, rescale: bool=True, print_desc: bool=False):
    ode = ODEBENCH[idx]
    if print_desc:
        print('index:', idx)
        print(ode['eq_description'])
        print('----')

    # clean and reformat strings
    exprs = ode['eq']
    p = re.findall(r'x_\d+', exprs)
    for pi in p:
        exprs = exprs.replace(pi, 'x'+str(int(pi[2:]) + 1)) # e.g. 'x_0' -> 'x1'

    exprs = exprs.split('|')
    exprs = [sy.simplify(e) for e in exprs]

    # get constant values
    const_vals = ode['consts'] # List[Lsit[float]]
    const_syms = sy.symbols([f'c_{i}' for i in range(len(const_vals[0]))])

    # ode instance
    exprs_list = []
    for const in const_vals:
        exprs_list.append([expr.subs(dict(zip(const_syms, const))) for expr in exprs])

    # integrate
    trajs_list = []
    sequence_list = []
    dgen = DatasetGenerator(None)
    _options = deepcopy(OPTIONS)
    t_span = _options.pop('t_span')
    t_eval= _options.pop('t_eval')
    x0s = ode['init']

    for exprs in exprs_list:
        sequence = SymbolicExpression().sympy_to_sequence(exprs)
        sequence_list.append(sequence)
        ode_fn = dgen.get_ode_func_from_sequence(sequence)
        trajs = []
        for x0 in x0s:
            res = solve_ivp(ode_fn, t_span, x0, t_eval=t_eval, **_options)
            traj = np.concatenate([res.t[:,None], res.y.T], axis=1) # [N,D+1]
            trajs.append(traj)

        # add noise and subsample
        if noise_sig > 0: trajs = add_noise_to_trajs(trajs, noise_sig, deterministic=True)
        if drop_rate > 0: trajs = subsample_trajs(trajs, drop_rate, deterministic=True)

        # rescale trajs
        if rescale:
            trajs,_ = rescale_trajs(trajs, X0_RADIUS, [TIME_RANGR_MIN, TIME_RANGR_MAX])

        trajs_list.append(trajs)
    
    return trajs_list, sequence_list

def plot_odebench(idx, rescale=False, print_desc=False, figsize=None, ncols=3, **kwargs):
    trajs_list, sequence_list = solve_odebench(idx, noise_sig=0., drop_rate=0., rescale=rescale, print_desc=print_desc)

    figs = []
    # for different ode instances (diferent constant values)
    for trajs, seq in zip(trajs_list, sequence_list):
        fig,axs = get_grid(len(trajs), ncols=ncols, figsize=figsize, return_fig=True)
        figs.append(fig)
        SymbolicExpression().print_readable_sequence(seq)

        # for different initial conditions
        for traj,ax in zip(trajs, axs):
            plot_trajs(traj[:,0], traj[:,1:], ax=ax, **kwargs)

        fig.tight_layout(pad=0.9)
        plt.show()
    return figs
