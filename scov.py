from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import time 
from joblib import Parallel, delayed
import itertools
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from numpy.linalg import svd
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.utils.extmath import randomized_svd
# import rioxarray as rxr
# from rasterio import CRS
import scipy.spatial
from scipy.spatial import cKDTree
from scipy.linalg import solve
import scipy.sparse as sp
from scipy.interpolate import BSpline
from sklearn.cluster import KMeans
from sksparse.cholmod import cholesky
#
import utils
from tqdm import tqdm
from threadpoolctl import threadpool_limits

class SpatialCovariance:

	def __init__(self, data, nonstationary=True, n_components = 1, max_lag=30, block_sz = 5, n_blocks = 20, n_cores = -1):
		self.data = self._standardize_input(data).compute()
		self.nonstationary = nonstationary
		self.max_lag = max_lag
		self.block_sz = block_sz
		self.n_blocks = n_blocks
		self.n_components = n_components
		self.n_cores = n_cores
		#
		self.holdout_ = None
		self.holdout_centers_ = None
		self.holdout_ids_ = None
		self.spatcov_ = None
		self.alpha_lat_ = None 
		self.alpha_lon_ = None 
		self.alpha_rot_ = None
		self.t_u_ = None
		self.t_v_ = None
		self.lam_lat_ = None
		self.lam_lon_ = None
		self.sill_ = None
		self.ls1_ = None
		self.ls2_ = None
		#
		self.generate_holdout()

	def _standardize_input(self, data):
		if not isinstance(data, (xr.DataArray, xr.Dataset)):
			raise TypeError("Input must be an xarray DataArray or Dataset.")
		
		if 'location' in data.dims and not isinstance(data.indexes.get('location'), pd.MultiIndex):
			if 'lat' in data.coords and 'lon' in data.coords:
				return data.set_index(location=['lat', 'lon'])

		if 'lat' in data.dims and 'lon' in data.dims:
			return data.stack(location=['lat', 'lon'])

		if 'location' in data.dims and isinstance(data.indexes.get('location'), pd.MultiIndex):
			return data
		    
		raise ValueError("Data must contain 'lat' and 'lon' coordinates or dimensions.")

	def generate_holdout(self):
		#
		self.holdout_, self.holdout_ids_, self.holdout_centers_ = utils._create_mask(self.data,n_blocks=self.n_blocks,block_sz=self.block_sz)
		#
		#return self
	
	def plot_holdout(self):
		M_da = xr.DataArray(
		    data=(self.holdout_!=0).numpy().astype(float),
		    dims=('location'),
		    coords={
			        'location': self.data.location
		    }
		)
		M_da = M_da.unstack()
		M_da = M_da.sortby(['lon','lat'])
		M_da.plot(add_colorbar=False)
		plt.show()

	def plot_nonstationary_spatcov(self):
		if not self.nonstationary:
			raise ValueError(
				"This plot can only be used when nonstationary=True. "
				"The current model is initialized as stationary."
			)

		if not hasattr(self, 'lam_lat_'):
			raise ValueError("plot_nonstationary_spatcov can only be used after .fit() is completed.")

		# --- Data Preparation ---
		def prep_da(data):
			da = xr.DataArray(data=data, dims=('location'), coords={'location': self.data.location})
			return da.unstack().sortby('lon', 'lat')

		W_da = prep_da(self.lam_lat_)
		W_da2 = prep_da(self.lam_lon_)
		W_da3 = prep_da(self.rot_)
		
		ar = np.log(np.maximum(self.lam_lat_, self.lam_lon_) / (np.minimum(self.lam_lat_, self.lam_lon_) + 1e-8))
		W_da4 = prep_da(ar)

		# --- Plotting Configuration ---
		title_size = 24
		label_size = 14
		fig, axes = plt.subplots(1, 4, figsize=(22, 6)) # Slightly larger fig for larger text
		
		# Helper to apply labels consistently
		def format_ax(ax, title, is_first=False):
			ax.set_title(title, fontsize=title_size, pad=10)
			ax.set_xlabel('', fontsize=label_size)
			ax.set_ylabel('' if is_first else '', fontsize=label_size)
			# Increase tick label size
			ax.tick_params(labelsize=10)

		# 1. Latitudinal
		W_da.plot(ax=axes[0], cmap=cm.plasma, #vmin=3, vmax=5, 
					add_colorbar=True, cbar_kwargs={'format': '%.1f'})
		format_ax(axes[0], 'Latitudinal length scale', is_first=True)
		
		# 2. Longitudinal
		W_da2.plot(ax=axes[1], cmap=cm.plasma, #vmin=3, vmax=5, 
					add_colorbar=True, cbar_kwargs={'format': '%.1f'})
		format_ax(axes[1], 'Longitudinal length scale')
		
		# 3. Rotation
		W_da3.plot(ax=axes[2], cmap=cm.plasma, vmin=0, 
					add_colorbar=True, cbar_kwargs={'format': '%.2f'})
		format_ax(axes[2], 'Rotation')
		
		# 4. Anisotropy
		W_da4.plot(ax=axes[3], cmap=cm.plasma, add_colorbar=True, 
					cbar_kwargs={'format': '%.1f'})
		format_ax(axes[3], 'Anisotropy log ratio')
		
		plt.tight_layout()
		return fig

	def _evaluate_ls(self, ls1, ls2, rot, init_phi):
		Sigma = utils._compute_spat_cov_rs(
			self.data, 
			phi=init_phi, 
			length_scale=ls1, 
			length_scale2=ls2, 
			rot = rot,
			max_lag=self.max_lag
		)        
		U, L, Ez, sigma2, loss, loss_tot = utils._cv_spatPCA(
			self.data, 
			Sigma, 
			self.holdout_, 
			k=self.n_components, 
			phi=init_phi
		)
		if self.nonstationary:
			return (ls1, ls2, rot, loss)
		else:
			return (ls1, ls2, rot, loss_tot)

	def _evaluate_phi(self, phi):
		Sigma = utils._compute_spat_cov_rs(
			self.data, 
			phi=1, 
			length_scale=5,
			max_lag=self.max_lag
		)        
		#
		U, L, Ez, sigma2, loss, loss_tot = utils._cv_spatPCA(
			self.data, 
			phi*Sigma, 
			self.holdout_, 
			k=self.n_components, 
			phi=phi
		)
		
		return (phi, loss_tot)

	def _evaluate_phi_post(self, phi):
		Sigma = self.spatcov_
		#
		U, L, Ez, sigma2, loss, loss_tot = utils._cv_spatPCA(
			self.data, 
			phi*Sigma, 
			self.holdout_, 
			k=self.n_components, 
			phi=phi
		)
		
		return (phi, loss_tot)

	def fit(self, lss=[1,3,5,7], phis=[1e3,5e3,1e4,5e4], rots=[0, 1*np.pi/16, 2*np.pi/16, 3*np.pi/16]):
		#
		if self.nonstationary:
			self._fit_nonstationary(lss, phis, rots)
		else:
			self._fit_stationary(lss, phis)
			
	def _fit_nonstationary(self, lss, phis, rots):
		#
		#
		#fit best phi for fixed ls
		#		
		if len(phis)>1:
			results = list(tqdm(
				Parallel(n_jobs=self.n_cores, return_as="generator")(
					delayed(self._evaluate_phi)(phi) 
					for phi in phis
				),
				total = len(phis),
				desc = 'Validating sill'
			))
			#
			self.sill_, _ = min(results, key=lambda x: x[1])
		else:
			self.sill_ = phis[0]
		#
		print(f'The prior sill estimate is {self.sill_}')
		#fit best lss
		#
		ls_combinations = list(itertools.product(lss, lss, rots))
		init_phi = self.sill_
		#
		results = list(tqdm(
			Parallel(n_jobs=self.n_cores, return_as='generator')(
				delayed(self._evaluate_ls)(ls1, ls2, rot, init_phi) 
				for ls1, ls2, rot in ls_combinations
			), 
			total=len(ls_combinations),
			desc='Validating length scales'
		))

		res_dict = {(res[0], res[1], res[2]): res[3] for res in results}
		loss_df = pd.DataFrame(res_dict)

		loss_df.index = self.holdout_ids_

		minimizer = loss_df.idxmin(axis=1).rename('ls_hat')
		
		centers_df = pd.DataFrame(
			self.holdout_centers_,
			index=self.holdout_ids_,
			columns=['mean_lat', 'mean_lon']
		)
		
		final_results = pd.concat([centers_df, minimizer], axis=1).dropna()
		
		self.minimizer_ = np.column_stack([
			np.array(final_results['ls_hat'].tolist()),
			final_results['mean_lat'].values,
			final_results['mean_lon'].values
		])
		print("Fitting spline to the optimal length scales...")
		self.alpha_lat_, self.alpha_lon_, self.alpha_rot_, self.t_u_, self.t_v_ = utils._fit_spline(self.data,self.minimizer_)
		#fit with variance=1 so you only have to run it once and can instead adjust it by multiplying phi
		self.spatcov_, self.lam_lat_, self.lam_lon_, self.rot_ = utils._construct_nonstat_cov(self.data.lat.values, self.data.lon.values, self.alpha_lat_, self.alpha_lon_, self.alpha_rot_, self.t_u_, self.t_v_, variance=1, max_lag=self.max_lag)
		#
		#
		self.spatcov_ = self.sill_ * self.spatcov_

	def _fit_stationary(self, lss, phis):
		#
		#fit best ls for fixed init_phi
		#
		ls_combinations = list(itertools.product(lss, lss))
		init_phi = 1e4
		
		results = list(tqdm(
			Parallel(n_jobs=self.n_cores, return_as='generator')(
				delayed(self._evaluate_ls)(ls1, ls2, [0], init_phi) 
				for ls1, ls2 in ls_combinations
			),
			total = len(ls_combinations),
			desc = 'Validating length scales'
		))
		
		self.ls1_, self.ls2_, _, _ = min(results, key=lambda x: x[2])
		self.spatcov_ = utils._compute_spat_cov_rs(self.data, 1, self.ls1_, self.ls2_, max_lag=self.max_lag)
		#
		#fit best phi for fixed ls
		#		
		results = list(tqdm(
			Parallel(n_jobs=self.n_cores,return_as='generator')(
				delayed(self._evaluate_phi)(phi) 
				for phi in phis
			),
			total = len(phis),
			desc = 'Validating sill'
		))

		self.sill_, _ = min(results, key=lambda x: x[1])
		self.spatcov_ = self.sill_ * self.spatcov_

