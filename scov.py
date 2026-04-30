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

	def __init__(self, data, nonstationary=True,n_components = 1,rotation=True, max_lag=30, blk_sz = 5, n_clusters = 20):
		self.data = self._standardize_input(data)
		self.nonstationary = nonstationary
		self.rotation = rotation
		self.max_lag = max_lag
		self.blk_sz = blk_sz
		self.n_clusters = n_clusters
		self.n_components = n_components
		#
		self.holdout_ = None
		self.holdout_centers_ = None
		self.holdout_ids_ = None
		self.spatcov_ = None
		self.alpha_lat_ = None 
		self.alpha_lon_ = None 
		self.t_u_ = None
		self.t_v_ = None
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
		self.holdout_, self.holdout_ids_, self.holdout_centers_ = utils._create_mask(self.data,n_clusters=self.n_clusters,blk_sz=self.blk_sz)
		#
		#return self
	
	def plot_holdout(self):
		M_da = xr.DataArray(
		    data=(self.holdout_!=0),
		    dims=('location'),
		    coords={
			        'location': self.data.location
		    }
		)
		M_da = M_da.unstack()
		M_da = M_da.sortby('lon','lat')
		M_da.plot(add_colorbar=False)
		plt.show()

	def plot_nonstationary_spatcov(self):
		if not self.nonstationary:
			raise ValueError(
				"This plot can only be used when nonstationary=True. "
				"The current model is initialized as stationary."
			)

		if not hasattr(self, 'alpha_lat_'):
			raise ValueError("plot_nonstationary_spatcov can only be used after .fit() is completed.")
		degree = 3
		lats = self.data.lat.values #optimal_ls_array[:, 2]
		lons = self.data.lon.values #optimal_ls_array[:, 3]
		N = len(lats)

		B_u = BSpline.design_matrix(lats, self.t_u_, degree).toarray()
		B_v = BSpline.design_matrix(lons, self.t_v_, degree).toarray()

		# Tensor product to get the 2D basis, then predict and exponentiate
		B = np.einsum('ik,il->ikl', B_v, B_u).reshape(N, -1)
		lam_lat = np.exp(B @ self.alpha_lat_)
		lam_lon = np.exp(B @ self.alpha_lon_)

		W_da = xr.DataArray(
		    data=lam_lat,
		    dims=('location'),
		    coords={
		        'location': self.data.location
		    }
		)
		W_da = W_da.unstack()
		W_da = W_da.sortby('lon','lat')

		W_da2 = xr.DataArray(
			data=lam_lon,
			dims=('location'),
			coords={
				'location': self.data.location
			}
		)
		W_da2 = W_da2.unstack()
		W_da2 = W_da2.sortby('lon','lat')

		fig, axes = plt.subplots(1, 2, figsize=(10, 4))

		W_da.plot(ax=axes[0], cmap='copper', add_colorbar=True, cbar_kwargs={'format': '%.1f'})
		axes[0].set_title('Latitudinal length scale')
		axes[0].set_xlabel('Longitude')
		axes[0].set_ylabel('Latitude')

		W_da2.plot(ax=axes[1], cmap='copper', add_colorbar=True, cbar_kwargs={'format': '%.1f'})
		axes[1].set_title('Longitudinal length scale')
		axes[1].set_xlabel('Longitude')
		axes[1].set_ylabel('Latitude')

		plt.tight_layout()
		plt.show()

	def _evaluate_ls(self, ls1, ls2, init_phi):
		Sigma = utils._compute_spat_cov_rs(
			self.data, 
			phi=init_phi, 
			length_scale=ls1, 
			length_scale2=ls2, 
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
			return (ls1, ls2, loss)
		else:
			return (ls1, ls2, loss_tot)

	def _evaluate_phi(self, phi):
		U, L, Ez, sigma2, loss, loss_tot = utils._cv_spatPCA(
			self.data, 
			phi*self.spatcov_, 
			self.holdout_, 
			k=self.n_components, 
			phi=phi
		)
		
		return (phi, loss_tot)

	def fit(self, lss=[1,3,5,7,9], phis=[1e1,1e2,1e3,1e4,1e5]):			
		#
		if self.nonstationary:
			self._fit_nonstationary(lss,phis)
		else:
			self._fit_stationary(lss,phis)
			
	def _fit_nonstationary(self, lss, phis):
		#
		#fit best ls for fixed init_phi
		#
		start_time = time.time()
		print(f"[{time.strftime('%H:%M:%S')}] Initializing fit...")
		ls_combinations = list(itertools.product(lss, lss))
		init_phi = 1e7
		# all_cores = 10
		# workers = 5
		# math_workers = all_cores // workers
		# torch.set_num_threads(math_workers)
		# with threadpool_limits(limits=math_workers, user_api='blas'):
		# 	with threadpool_limits(limits=math_workers, user_api='openmp'):
		results = list(tqdm(
			Parallel(n_jobs=32, return_as='generator')(
				delayed(self._evaluate_ls)(ls1, ls2, init_phi) 
				for ls1, ls2 in ls_combinations
			), 
			total=len(ls_combinations),
			desc='Validating length scales for prior spatial covariance.'
		))

		res_dict = {(res[0], res[1]): res[2] for res in results}
		loss_df = pd.DataFrame(res_dict)
		
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
		print("Fitting spline to the optimal length scales at held-out locations...")
		self.alpha_lat_, self.alpha_lon_, self.t_u_, self.t_v_ = utils._fit_spline(self.data,self.minimizer_)
		#fit with variance=1 so you only have to run it once and can instead adjust it by multiplying phi
		self.spatcov_ = utils._construct_nonstat_cov(self.data.lat.values, self.data.lon.values, self.alpha_lat_, self.alpha_lon_, self.t_u_, self.t_v_, variance=1, max_lag=self.max_lag)
		#
		#fit best phi for fixed ls
		#		
		results = list(tqdm(
			Parallel(n_jobs=-1, return_as="generator")(
				delayed(self._evaluate_phi)(phi) 
				for phi in phis
			),
			total = len(phis),
			desc = 'Validating sill variance.'
		))

		# results = Parallel(n_jobs=-1)(
		# 	delayed(self._evaluate_phi)(phi) 
		# 	for phi in tqdm(phis, desc='Validating sill variance for prior spatial covariance.')
		# )

		self.sill_, _ = min(results, key=lambda x: x[1])
		self.spatcov_ = self.sill_ * self.spatcov_
		end_time = time.time()
		print(f"[{time.strftime('%H:%M:%S')}] Fit complete in {end_time - start_time:.2f} seconds.")

	def _fit_stationary(self, lss, phis):
		#
		#fit best ls for fixed init_phi
		#
		ls_combinations = list(itertools.product(lss, lss))
		init_phi = 1e5 
		
		results = list(tqdm(
			Parallel(n_jobs=-1, return_as='generator')(
				delayed(self._evaluate_ls)(ls1, ls2, init_phi) 
				for ls1, ls2 in ls_combinations
			),
			total = len(ls_combinations),
			desc = 'Validating length scales for prior spatial covariance.'
		))
		
		self.ls1_, self.ls2_, _ = min(results, key=lambda x: x[2])
		self.spatcov_ = utils._compute_spat_cov_rs(self.data, 1, self.ls1_, self.ls2_, max_lag=self.max_lag)
		#
		#fit best phi for fixed ls
		#		
		results = list(tqdm(
			Parallel(n_jobs=-1,return_as='generator')(
				delayed(self._evaluate_phi)(phi) 
				for phi in phis
			),
			total = len(phis),
			desc = 'Validating sill variance.'
		))

		self.sill_, _ = min(results, key=lambda x: x[1])
		self.spatcov_ = self.sill_ * self.spatcov_











