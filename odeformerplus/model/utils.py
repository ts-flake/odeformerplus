import torch
import numpy as np
import random, os
from tqdm.auto import tqdm
from typing import List
from scipy.integrate import solve_ivp
from odeformerplus.envs.symbolic import SymbolicExpression
from odeformerplus.envs.generators import DatasetGenerator
from odeformerplus.envs.dataset import plot_trajs, get_grid, rescale_trajs
from odeformerplus.envs.utils import load_config
from odeformerplus.envs.utils import suppress_output

config = load_config('odeformerplus/envs/config.yaml')

MAX_XDIM = config.expression_tree.max_xdim
TIME_RANGR_MIN = config.dataset_generator.traj_time_range_l
TIME_RANGR_MAX = config.dataset_generator.traj_time_range_u
X0_RADIUS = config.dataset_generator.x0_range_traj


def count_params(m):
    "Count trainable parameters."
    count_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    count_total = sum(p.numel() for p in m.parameters())
    print(f'{m.__class__.__name__} has {count_trainable/1e6:.2f}M trainable parameters out of {count_total/1e6:.2f}M total parameters.')
    return count_trainable

def reset_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device
    
def freeze_params(m):
    for param in m.parameters():
        param.requires_grad = False

def unfreeze_params(m):
    for param in m.parameters():
        param.requires_grad = True

####################
## TRAINING UTILS ##
####################

def get_learning_rate(step, warmup_steps=4000, d_model=256):
    if step == 0:
        step = 1
    return d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

def get_lr_scheduler(optimizer, d_model=256, warmup_steps=4000):
    print('Learning rate scheduler:')
    print('------------------------')
    print(f'initial lr: {get_learning_rate(0, warmup_steps, d_model):.3e}')
    print(f'peak lr: {get_learning_rate(warmup_steps, warmup_steps, d_model):.3e}')
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: get_learning_rate(step, warmup_steps, d_model))

def save_ckpt(model, path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save checkpoint   
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

def load_ckpt(model, path):
    ckpt = torch.load(path, weights_only=True)
    msg = model.load_state_dict(ckpt['model_state_dict'])
    print(msg)

################
## EVAL UTILS ##
################
from odeformerplus.envs.utils import load_config

_CONFIG = load_config('odeformerplus/envs/config.yaml')
OPTIONS = _CONFIG.dataset_generator.solver_options

def calculate_R2(pred_trajs: List[np.ndarray], targ_trajs: List[np.ndarray]):
    r2s = []
    for pred,targ in zip(pred_trajs, targ_trajs):
        residual = pred[:,1:] - targ[:,1:]
        variance = targ[:,1:] - np.mean(targ[:,1:], axis=0)[None,:]
        r2s.append(1 - np.sum(residual**2) / np.sum(variance**2))
    return r2s

def calculate_R2_accuracy(pred_trajs: List[np.ndarray], targ_trajs: List[np.ndarray], threshold=0.9):
    r2s = calculate_R2(pred_trajs, targ_trajs)
    return sum(np.array(r2s) >= threshold) / len(r2s)

def integrate_ode(sequence: List[str], x0s: List[np.ndarray], t_evals: List[np.ndarray], solver_options: dict=None):
    if solver_options is None:
        solver_options = OPTIONS.to_dict()

    # Integrate ODE
    dgen = DatasetGenerator(None)
    ode_func = dgen.get_ode_func_from_sequence(sequence)
    ode_trajs = []
    for x0,t_eval in zip(x0s, t_evals):
        t_span = (min(t_eval), max(t_eval))
        with suppress_output():
            res = solve_ivp(ode_func, t_span, x0, t_eval=t_eval, **solver_options)
        ode_traj = np.concatenate([res.t[:,None], res.y.T], axis=1)
        ode_trajs.append(ode_traj)
    return ode_trajs

def integrate_ode_and_plot(
        sequence: List[str],
        input_trajs: List[np.ndarray],
        rescale_input: bool=True,
        solver_options: dict=None,
        calc_r2_score: bool=True,
        r2_threshold: float=0.9
    ):
    if solver_options is None:
        solver_options = OPTIONS.to_dict()

    # Print ODE
    print('ODEs')
    print('----')
    SymbolicExpression().print_readable_sequence(sequence)
    print('----')
    
    # Rescale input
    if rescale_input:
        input_trajs, info = rescale_trajs(input_trajs, X0_RADIUS, [TIME_RANGR_MIN, TIME_RANGR_MAX])

    # Integrate ODE
    ode_trajs = integrate_ode(sequence, [traj[0,1:] for traj in input_trajs], [traj[:,0] for traj in input_trajs], solver_options)

    # R2 accuracy
    if calc_r2_score:
        r2_acc = calculate_R2_accuracy(ode_trajs, input_trajs, r2_threshold)
        print(f'Average R2 accuracy (threshold={r2_threshold}): {r2_acc:.3f}')

    # Plot
    fig,axs = get_grid(len(input_trajs), ncols=3, return_fig=True)
    for ax,input_traj,ode_traj in zip(axs,input_trajs,ode_trajs):
        plot_trajs(input_traj[:,0], input_traj[:,1:], ax=ax, ms=8, alpha=.2, marker='.', linestyle='')
        plot_trajs(ode_traj[:,0], ode_traj[:,1:], ax=ax)
    fig.tight_layout();
    return r2_acc if calc_r2_score else None


def eval_odebench(inference_fn, noise_sig=0.05, drop_rate=0.2, solver_options: dict=None, r2_threshold: float=0.9):
    """
    Evaluate ODEBench using a given inference function.

    inference_fn: function that takes in List[np.ndarray] and List[str], and returns List[str].
    """
    from odeformerplus.odebench.utils import ODEBENCH, solve_odebench
    res = []
    success_count = 0
    failed_count = 0
    ave_r2 = []
    for idx in tqdm(range(len(ODEBENCH)), desc='Evaluating ODEBench'):
        trajs_list, sequence_list = solve_odebench(idx, noise_sig=noise_sig, drop_rate=drop_rate, rescale=True, print_desc=False)
        r2_accs = []
        preds = []
        for trajs, sequence in zip(trajs_list, sequence_list):
            pred = inference_fn(trajs, sequence)
            preds.append(pred)
            try:
                ode_trajs = integrate_ode(pred, [traj[0,1:] for traj in trajs], [traj[:,0] for traj in trajs], solver_options)
            except Exception:
                print(f'{ODEBENCH[idx]["id"]}: Error in integrating predicted ODE')
                r2_accs.append(0)
                failed_count += 1
                continue
            try:
                r2_acc = calculate_R2_accuracy(ode_trajs, trajs, r2_threshold)
            except Exception:
                print(f'{ODEBENCH[idx]["id"]}: Error in calculating R2 score')
                r2_accs.append(0)
                failed_count += 1
                continue
            r2_accs.append(r2_acc)
            success_count += 1
        res.append({
            'id': ODEBENCH[idx]['id'],
            'eq_description': ODEBENCH[idx]['eq_description'],
            'odes': sequence_list,
            'r2_accs': r2_accs,
            'ave_r2_acc': np.mean(r2_accs),
            'preds': preds,
        })
        ave_r2.append(np.mean(r2_accs))
    
    print('--------------------------------')
    print(f'Success count: {success_count} / {success_count + failed_count}')
    print(f'Average R2 accuracy: {np.mean(ave_r2):.3f}')
    return res, success_count, np.mean(ave_r2)
