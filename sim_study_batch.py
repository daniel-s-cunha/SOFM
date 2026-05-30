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

# Update this to your actual SCC path (e.g., /projectnb/your_project/dcunha/SOFM)
repo_path = os.path.expanduser('/projectnb/modislc/users/danc/SOFM') 

if repo_path not in sys.path:
    sys.path.append(repo_path)

import utils
import sofm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    args = parser.parse_args()

    sample_sizes = [200, 500, 1000, 5000, 10000]
    sigmas = [5,10]
    max_lags = [2,20]
    prior_covs = [0,1,2]#[0]#["circle","constant","gradient"]
    growings = [True,False]
    replicates = [j for j in range(20)]
    #
    param_grid = list(itertools.product(sample_sizes, sigmas, max_lags, prior_covs, growings, replicates))
    #
    if args.task_id > len(param_grid):
        print(f"Task ID {args.task_id} out of range for grid of size {len(param_grid)}")
        return

    n_samples, sigma, max_lag, prior_cov, growing, replicate = param_grid[args.task_id - 1]
    print(f"Running Task {args.task_id}: n_samples = {n_samples}, sigma={sigma}, max_lag={max_lag}, prior_cov={prior_cov}, growing={growing}, replicate={replicate}")

    torch.set_num_threads(1)
    prior_covs_n = ["circle","constant","gradient"]#['independent'] #["circle","constant","gradient"]
    try:
        sq_len = 20
        print(f"\n{'='*40}")
        print(f"Running Sample Size: N = {n_samples}")
        print(f"{'='*40}")
        if growing:
            sq_len = 50#int(n_samples**0.5)
        
        start_time = time.time()
        
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
        
        # 2. Initialize and Fit Model
        # Note: Ensure block_sz matches your init parameter name (block_sz vs block_sz)
        mod = sofm.SOFM(da, n_components=3, max_lag = max_lag, nonstationary=True, block_sz=1, n_blocks=int(sq_len**2/10),n_cores=1)
        mod.fit(lss = [1,3,5],phis=[10**j for j in range(5)]+[5*10**j for j in range(5)],rots=[0])#[0,np.pi/8,2*np.pi/8,3*np.pi/8])
        sc = mod.spatcov_
        if prior_covs_n[prior_cov]=='independent':
            lat_mse = 0
            lon_mse = 0
            rot_mse = 0
        else:
            lat_mse = np.mean((sc.lam_lat_ - lam_lat)**2)
            lon_mse = np.mean((sc.lam_lon_ - lam_lon)**2)
            rot_mse = np.mean((sc.rot_ - rot)**2)
        # 3. Calculate Loss for U (Accounting for Sign Ambiguity)
        U_est = mod.U_.detach().numpy() if torch.is_tensor(mod.U_) else mod.U_
        U_true_np = U_true.detach().numpy() if torch.is_tensor(U_true) else U_true
        
        U_loss = 0.0
        for k in range(U_est.shape[1]):
            # Check MSE for both standard and flipped signs
            mse_plus = np.mean((U_est[:, k] - U_true_np[:, k])**2)
            mse_minus = np.mean((U_est[:, k] + U_true_np[:, k])**2)
            U_loss += min(mse_plus, mse_minus)
        U_loss /= U_est.shape[1] # Average MSE across the k components
        
        # 4. Calculate Loss for L
        L_est = mod.L_.detach().numpy() if torch.is_tensor(mod.L_) else mod.L_
        L_true_np = L_true.detach().numpy() if torch.is_tensor(L_true) else L_true
        L_loss = np.mean((L_est - L_true_np)**2)
        
        # 5. Calculate Loss for Sigma
        # Assuming mod.sigma2_ is variance, we take sqrt to compare to sigma_true
        sigma_est = np.sqrt(mod.sigma2_) if not torch.is_tensor(mod.sigma2_) else torch.sqrt(mod.sigma2_).item()
        sigma_true_val = sigma_true.item() if torch.is_tensor(sigma_true) else sigma_true
        sigma_loss = (sigma_est - sigma_true_val)**2
        
        run_time = time.time() - start_time
    except Exception as e:
        print(f"Error in computation: {e}")
        U_loss = -999
        L_loss = -999
        sigma_loss = -999
        lat_mse = -999
        lon_mse = -999
        rot_mse = -999
    #
    results = np.array([[n_samples, sigma, max_lag, prior_cov, growing, replicate, U_loss, L_loss, sigma_loss, lat_mse, lon_mse, rot_mse]])
    out_dir = '/projectnb/modislc/users/danc/SOFM/sofm_sim_study_gamma1e4/'
    os.makedirs(out_dir, exist_ok=True)
    
    filename = f'results_n_samples{n_samples}_sigma{sigma}_max_lag{max_lag}_prior_cov{prior_cov}_growing{growing}_replicate{replicate}_nonstat_v0.csv'
    np.savetxt(os.path.join(out_dir, filename), results, delimiter=',')
    print(f"Task complete. Saved to {filename}")

if __name__ == "__main__":
    main()


