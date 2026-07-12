import torch
import numpy as np
import itertools
import os
import sys
import xarray as xr
import argparse
import xeofs as xe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time

repo_path = os.path.expanduser('/projectnb/modislc/users/danc/SOFM') 

if repo_path not in sys.path:
    sys.path.append(repo_path)

import utils
import sofm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    args = parser.parse_args()

    sample_sizes = [3000]
    sigmas = [5]
    max_lags = [20]
    prior_covs = [2]#[0]#["circle","constant","gradient"]
    sq_lens = [10,20,30,40,50,60,70,80,90,100]
    replicates = [j for j in range(20)]
    #
    param_grid = list(itertools.product(sample_sizes, sigmas, max_lags, prior_covs, sq_lens, replicates))
    #
    if args.task_id > len(param_grid):
        print(f"Task ID {args.task_id} out of range for grid of size {len(param_grid)}")
        return

    n_samples, sigma, max_lag, prior_cov, sq_len, replicate = param_grid[args.task_id - 1]
    print(f"Running Task {args.task_id}: n_samples = {n_samples}, sigma={sigma}, max_lag={max_lag}, prior_cov={prior_cov}, sq_len={sq_len}, replicate={replicate}")

    torch.set_num_threads(1)
    prior_covs_n = ["circle","constant","gradient"]#['independent'] #["circle","constant","gradient"]
    try:
        print(f"\n{'='*40}")
        print(f"Running Sample Size: N = {n_samples}")
        print(f"{'='*40}")
        
        # 1. Generate Data
        da, lam_lat, lam_lon, rot, U_true, L_true, sigma_true = utils._gen_spat_da(
            sq_len=sq_len,
            k=3, 
            n_samples=n_samples, 
            max_lag=20, 
            ls1=1, 
            ls2=1, 
            mode=prior_covs_n[prior_cov],
            sigma=sigma
        )
        
        start_time = time.time()
        mod = sofm.SOFM(da, n_components=3, max_lag = max_lag, nonstationary=True, block_sz=1, n_blocks=int(sq_len**2/10),n_cores=1)
        mod.fit(lss = [1,3,5],phis=[10**j for j in range(5)]+[5*10**j for j in range(5)],rots=[0])#[0,np.pi/8,2*np.pi/8,3*np.pi/8])
        run_time = time.time() - start_time
        
    except Exception as e:
        print(f"Error in computation: {e}")
        run_time = -999
    #
    results = np.array([[n_samples, sigma, max_lag, prior_cov, sq_len, replicate, run_time]])
    out_dir = '/projectnb/modislc/users/danc/SOFM/sofm_sim_study_fixphi/'
    os.makedirs(out_dir, exist_ok=True)
    
    filename = f'compTime_sq_len{sq_len}_replicate{replicate}_v0.csv'
    np.savetxt(os.path.join(out_dir, filename), results, delimiter=',')
    print(f"Task complete. Saved to {filename}")

if __name__ == "__main__":
    main()


