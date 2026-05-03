import utils
from scov import SpatialCovariance
import time
import torch
import numpy as np

class SOFM:
    def __init__(
        self, 
        data, 
        n_components=1, 
        nonstationary=True, 
        max_lag=20, 
        block_sz=5, 
        n_blocks=20,
        n_cores = -1
    ):
        self.spatcov_ = SpatialCovariance(
            data=data,
            nonstationary=nonstationary,
            n_components=n_components,
            max_lag=max_lag,
            block_sz=block_sz,
            n_blocks=n_blocks,
            n_cores = n_cores
        )
        
        self.n_components = n_components
        #
        self.U_ = None
        self.L_ = None
        self.Ez_ = None
        self.sigma2_ = None

    def fit(self, lss=[0.5, 1, 3, 5], phis=[1e2,5e2,1e3,5e3,1e4,5e4,1e5]):
        #
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Estimating prior spatial covariance...")
        self.spatcov_.fit(lss=lss, phis=phis)
        
        Sigma = self.spatcov_.spatcov_
        phi = self.spatcov_.sill_
        data = self.spatcov_.data
        
        print("Refitting full model...")
        
        U, L, Ez, sigma2, loss, = utils._spatPCA(
            Y_da=data,
            Sigma=Sigma,
            k=self.n_components,
            phi=phi
        )
        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Fit complete in {end_time - start_time:.2f} seconds.")
        self.U_ = U
        self.L_ = L.detach()
        self.Ez_ = Ez
        self.sigma2_ = sigma2.detach()
