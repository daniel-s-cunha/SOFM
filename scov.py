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


class SpatialCovariance:

	def __init__(self, data, nonstationary=True,rotation=True,max_lag=20, blk_sz = 5, n_clusters = 20):
		self.data = self._standardize_input(data)
		self.nonstationary = nonstationary
		self.rotation = rotation
		self.max_lag = max_lag
		self.blk_sz = blk_sz
		self.n_clusters = n_clusters
		#
		self.holdout_ = None
		self.holdout_centers_ = None
		self.holdout_ids_ = None
		self.spatcov_ = None
		self.alpha_lat = None 
		self.alpha_lon = None 
		self.t_u = None
		self.t_v = None
		self.sill = None
		self.ls1 = None
		self.ls2 = None

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
		    data=self.holdout_,
		    dims=('location'),
		    coords={
			        'location': self.data.location
		    }
		)
		M_da = M_da.unstack()
		M_da = M_da.sortby('lon','lat')
		M_da.plot(add_colorbar=False)
		plt.show()
		return self

	def _evaluate_ls_pair(self, ls1, ls2, init_phi):
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
			k=3, 
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
			k=3, 
			phi=phi
		)
		
		return (phi, loss_tot)

	

	def fit(self, lss=[1,3,5], phis=[1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12]):
		if self.nonstationary:
			self._fit_nonstationary(lss,phis)
		else:
			self._fit_stationary(lss,phis)
			
	def _fit_nonstationary(self, lss=[1,3,5], phis=[1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12]):
		#
		#fit best ls for fixed init_phi
		#
		ls_combinations = list(itertools.product(lss, lss))
		init_phi = 1e3 
		
		results = Parallel(n_jobs=-1)(
			delayed(self._evaluate_ls_pair)(ls1, ls2, init_phi) 
			for ls1, ls2 in ls_combinations
		)
		
		res_dict = {(res[0], res[1]): res[2] for res in results}
		loss_df = pd.DataFrame(res_dict)
		
		minimizer = loss_df.idxmin(axis=1).rename('ls_hat')
		
		centers_df = pd.DataFrame(
			self.holdout_centers_,
			index=self.holdout_ids_,
			columns=['mean_lat', 'mean_lon']
		)
		
		final_results = pd.concat([centers_df, minimizer], axis=1)
		
		self.minimizer_ = np.column_stack([
			np.array(final_results['ls_hat'].tolist()),
			final_results['mean_lat'].values,
			final_results['mean_lon'].values
		])
		
		self.alpha_lat, self.alpha_lon, self.t_u, self.t_v = utils._fit_spline(self.data,self.minimizer_)
		#fit with variance=1 so you only have to run it once and can instead adjust it by multiplying phi
		self.spatcov_ = utils._construct_nonstat_cov(self.data.lat.values, self.data.lon.values, self.alpha_lat, self.alpha_lon, self.t_u, self.t_v, variance=1, max_lag=self.max_lag)
		#
		#fit best phi for fixed ls
		#		
		results = Parallel(n_jobs=-1)(
			delayed(self._evaluate_phi)(phi) 
			for phi in phis
		)

		self.sill, _ = min(results, key=lambda x: x[1])
		self.spatcov_ = self.sill * self.spatcov_
		

	def _fit_stationary(self, lss=[1,3,5], phis=[1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12]):
		#
		#fit best ls for fixed init_phi
		#
		ls_combinations = list(itertools.product(lss, lss))
		init_phi = 1e3 
		
		results = Parallel(n_jobs=-1)(
			delayed(self._evaluate_ls_pair)(ls1, ls2, init_phi) 
			for ls1, ls2 in ls_combinations
		)
		
		self.ls1, self.ls2, _ = min(results, key=lambda x: x[2])
		self.spatcov_ = utils._compute_spat_cov_rs(self.data, 1, self.ls1, self.ls2, max_lag=self.max_lag)
		#
		#fit best phi for fixed ls
		#		
		results = Parallel(n_jobs=-1)(
			delayed(self._evaluate_phi)(phi) 
			for phi in phis
		)

		self.sill, _ = min(results, key=lambda x: x[1])
		self.spatcov_ = self.sill * self.spatcov_

#choose the best ls
#---Take nll from across models and choose ls that has smallest nll
#joblib: run cv_spatpca for all phi values
#rerun spatPCA with final model #make this optional






















