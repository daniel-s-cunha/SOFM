import matplotlib.pyplot as plt
import math 
import utils
from scov import SpatialCovariance
import time
import torch
import numpy as np
import xarray as xr 

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

    def fit(self, lss=[1,3,5], phis=[1e1,1e2,1e3], rots = [0, np.pi/12, 2*np.pi/12]):
        #
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Estimating prior spatial covariance...")
        self.spatcov_.fit(lss=lss, phis=phis, rots = rots)
        
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

    def fit_contrast(self,phi = 1e3,lambda_neg=1):
        print("Fitting...")
        start_time = time.time()
        data = self.spatcov_.data
        Omega = utils._construct_omega_matrix(data,lambda_neg = lambda_neg)
        U, L, Ez, sigma2, loss, = utils._spatPCA(
            Y_da=data,
            Sigma=phi*Omega,
            k=self.n_components,
            phi=phi
        )
        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Fit complete in {end_time - start_time:.2f} seconds.")
        self.U_ = U
        self.L_ = L.detach()
        self.Ez_ = Ez
        self.sigma2_ = sigma2.detach()

    def plot_loadings(self,invert=True, robust=False):
        k = self.U_.shape[1]
        
        W_da = xr.DataArray(
            data=self.U_,
            dims=('location', 'factor'),
            coords={
                'location': self.spatcov_.data.location,
                'factor': np.arange(k)
            }
        )
        # W_da = W_da.unstack()
        # W_da = W_da.sortby('lon', 'lat')
        W_da = W_da.set_index(location=['lat', 'lon']).unstack('location')
        W_da = W_da.sortby(['lat', 'lon'])

        ncols = k
        nrows = 1

        g = (1*W_da).plot(
            col='factor', 
            col_wrap=None,
            cmap='RdBu_r', 
            add_colorbar=False,
            figsize=(3 * ncols, 3.5),
            robust=robust
        )

        # Clean up formatting
        g.set_titles(template="")
        g.set_axis_labels("", "")
        if invert:
            for ax in g.axes.flat:
                ax.invert_yaxis()
        return g.fig


