from odeformerplus.envs.generators import DatasetGenerator
from odeformerplus.envs.utils import print_dataset_stats
from multiprocessing import Pool
import argparse,os,json, re, time
import numpy as np


def func(seed, num_samples, dir_path, file_name):
    DatasetGenerator(seed).generate_dataset(num_samples, dir_path, file_name, print_stats=False)

def main(args):
    pool = Pool(args.num_workers)
    nums = [args.num_samples_per_worker]*args.num_workers
    _seed = np.random.randint(10000) if args.seed is None else args.seed
    seeds = np.random.default_rng(_seed).choice(int(1e5),size=args.num_workers)
    _dir = args.dataset_dir.rstrip('/')+'/'
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    files = [f'data_{i:d}' for i in range(1, args.num_workers+1)]
    
    for s,n,f in zip(seeds, nums, files):
        pool.apply_async(func, args=(s, n, _dir,f), error_callback=print)
    print('waiting for all subprocesses done...')
    pool.close()
    pool.join()
    print('all subprrocesses done.\n')
    
    # -- merge all data
    data_all = []
    for file in os.listdir(_dir):
        if re.match(r'data_\d+.json', file):
            file_path = os.path.join(_dir, file)
            
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    data_all.extend(data)
                except json.JSONDecodeError as e:
                    print(f"error reading {file}: {e}")
    
    out_file = _dir+'data_all.json'
    with open(out_file, 'w') as f:
        json.dump(data_all, f)
    print('saved all data to', out_file)

    # -- merge all stats
    def merge_dict(src, tgt):
        "merge tgt dict with src dict"
        if not tgt: return src
        for k,v in tgt.items():
            if k not in src:
                src[k] = v
            elif type(v) in [int, float]:
                src[k] += v
            elif type(v) == list:
                src[k].extend(v)
            elif type(v) == dict:
                src[k] = merge_dict(src[k], v)
        return src
    
    stats_all = {}
    for file in os.listdir(_dir):
        if re.match(r'data_\d+_stats.json', file):
            file_path = os.path.join(_dir, file)
            
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    stats_all = merge_dict(stats_all, data)
                except json.JSONDecodeError as e:
                    print(f"error reading {file}: {e}")
    out_file = _dir+'stats_all.json'
    with open(out_file, 'w') as f:
        json.dump(stats_all, f)
    print('saved all stats to', out_file)
    print_dataset_stats(stats_all)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
    parser.add_argument( '--seed', type=int, default=None, help='seed')
    parser.add_argument('--dataset_dir', type=str, default='dataset/', help='dataset directory')
    parser.add_argument('--num_samples_per_worker', type=int, default=10, help='number of samples per worker')
    args = parser.parse_args()
    start = time.time()
    main(args)
    print('time spend:', time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time()-start)))