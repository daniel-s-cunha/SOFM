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
        max_lag=30, 
        block_sz=5, 
        n_blocks=20
    ):
        self.spatcov_ = SpatialCovariance(
            data=data,
            nonstationary=nonstationary,
            n_components=n_components,
            max_lag=max_lag,
            block_sz=block_sz,
            n_blocks=n_blocks
        )
        
        self.n_components = n_components
        self.U_ = None
        self.L_ = None
        self.Ez_ = None
        self.sigma2_ = None

    def fit(self, lss=[1, 3, 5, 7], phis=[1e1, 1e2, 1e3, 1e4, 1e5]):
        #        
        print("Estimating prior spatial covariance...")
        self.spatcov_.fit(lss=lss, phis=phis)
        
        Sigma = self.spatcov_.spatcov_
        phi = self.spatcov_.sill_
        std_data = self.spatcov_.data
        
        print("")
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Fitting full model...")
        
        U, L, Ez, sigma2, loss, = utils._spatPCA(
            Y_da=std_data,
            Sigma=Sigma,
            k=self.n_components,
            phi=phi
        )
        
        self.U_ = U
        self.L_ = L.detach()
        self.Ez_ = Ez
        self.sigma2_ = sigma2.detach()
        
        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Final MAP Estimation complete in {end_time - start_time:.2f} seconds.")
