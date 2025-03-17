import signal
from signal import SIGALRM,SIG_IGN
from contextlib import contextmanager

import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import os

@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr, even from C/Fortran extensions."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)  # Save current stdout
    old_stderr = os.dup(2)  # Save current stderr

    try:
        os.dup2(devnull, 1)  # Redirect stdout to /dev/null
        os.dup2(devnull, 2)  # Redirect stderr to /dev/null
        yield  # Execute the block inside `with`
    finally:
        os.dup2(old_stdout, 1)  # Restore stdout
        os.dup2(old_stderr, 2)  # Restore stderr
        os.close(devnull)  # Close /dev/null
        os.close(old_stdout)  # Close saved stdout
        os.close(old_stderr)  # Close saved stderr


@contextmanager
def timeout(time):
    def raise_timeout(signum, frame):
        raise TimeoutError('timeout')

    # Set the signal handler for SIGALRM, note: not working on Windows!
    signal.signal(SIGALRM, raise_timeout)
    # Schedule the alarm signal to be sent after 'time' seconds
    signal.alarm(time)

    try:
        yield
    finally:
        # Disable the alarm signal
        signal.alarm(0)
        # Reset the signal handler to default
        signal.signal(SIGALRM, SIG_IGN)

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

def load_config(config_file_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    config_file = Path(config_file_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


#############
## DATASET ##
#############

def sort_dict_key(d, reverse=False):
    return {k:v for k,v in sorted(d.items(), key=lambda o: o[0], reverse=reverse)}

def sort_dict_value(d, reverse=False):
    return {k:v for k,v in sorted(d.items(), key=lambda o: o[1], reverse=reverse)}

def print_dataset_stats(stats: dict):
    print('\n-- total number of samples:', stats['num_samples'])
    print('dim: count')
    print(''.join([f'{k}: {v}\n' for k,v in sort_dict_key(stats['dimension']).items()]))

    print('\n-- total number of trajectories:', stats['total_num_trajs'])
    print('dim: {num_trajs: count}')
    _dict = {k:sort_dict_key(v, reverse=True) for k,v in stats['num_trajs_per_dim'].items()}
    print(''.join([f'{k}: {dict(v)}\n' for k,v in sort_dict_key(_dict).items()]))

    print('-- unary operators')
    unary = stats['unary_operators']
    unary_total = sum(unary.values())
    print(f'name: count (percent), total: {unary_total}')
    print(''.join([f'{k:<5}: {v} ({v/unary_total:.3f})\n' for k,v in sort_dict_value(unary, reverse=True).items()]))
    
    print('-- binary operators')
    binary = stats['binary_operators']
    binary_total = sum(binary.values())
    print(f'name: count (percent), total: {binary_total}')
    print(''.join([f'{k}: {v} ({v/binary_total:.3f})\n' for k,v in sort_dict_value(binary, reverse=True).items()]))
    
    print('-- average complexity per dimension')
    print(f"{np.mean(stats['complexity_per_dim']):.3f}")


##############
## PLOTTING ##
##############

def get_grid(
    n:int, # Number of axes in the returned grid
    nrows:int=None, # Number of rows in the returned grid, defaulting to `int(math.sqrt(n))`
    ncols:int=None, # Number of columns in the returned grid, defaulting to `ceil(n/rows)` 
    figsize:tuple=None, # Width, height in inches of the returned figure
    double:bool=False, # Whether to double the number of columns and `n`
    title:str=None, # If passed, title set to the figure
    return_fig:bool=False, # Whether to return the figure created by `subplots`
    flatten:bool=True, # Whether to flatten the matplot axes such that they can be iterated over with a single loop
    **kwargs
): # Returns just `axs` by default, and (`fig`, `axs`) if `return_fig` is set to True
    "From `fastai`. Return a grid of `n` axes, `rows` by `cols`"
    if nrows:
        ncols = ncols or int(np.ceil(n/nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n/ncols))
    else:
        nrows = int(np.sqrt(n))
        ncols = int(np.ceil(n/nrows))
    if double: ncols*=2 ; n*=2
    w,h = 3,2
    if not figsize: figsize = (w*ncols, h*nrows)
    fig,axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if flatten: axs = [ax if i<n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    if title is not None: fig.suptitle(title, weight='bold', size=14)
    return (fig,axs) if return_fig else axs