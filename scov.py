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
		self.data = self._standardize_input(data)
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

		W_da = xr.DataArray(
		    data=self.lam_lat_,
		    dims=('location'),
		    coords={
		        'location': self.data.location
		    }
		)
		W_da = W_da.unstack()
		W_da = W_da.sortby('lon','lat')

		W_da2 = xr.DataArray(
			data=self.lam_lon_,
			dims=('location'),
			coords={
				'location': self.data.location
			}
		)
		W_da2 = W_da2.unstack()
		W_da2 = W_da2.sortby('lon','lat')

		W_da3 = xr.DataArray(
			data=self.rot_,
			dims=('location'),
			coords={
				'location': self.data.location
			}
		)
		W_da3 = W_da3.unstack()
		W_da3 = W_da3.sortby('lon','lat')

		ar = np.maximum(self.lam_lat_, self.lam_lon_) / (np.minimum(self.lam_lat_, self.lam_lon_) + 1e-8)
		W_da4 = xr.DataArray(
			data=ar,
			dims=('location'),
			coords={
				'location': self.data.location
			}
		)
		W_da4 = W_da4.unstack()
		W_da4 = W_da4.sortby('lon','lat')

		fig, axes = plt.subplots(2, 2, figsize=(12, 8))

		W_da.plot(ax=axes[0][0], cmap=cm.plasma, add_colorbar=True, robust=True,  cbar_kwargs={'format': '%.1f'})#robust=True,
		axes[0][0].set_title('Latitudinal length scale')
		axes[0][0].set_xlabel('Longitude')
		axes[0][0].set_ylabel('Latitude')

		W_da2.plot(ax=axes[0][1], cmap=cm.plasma, add_colorbar=True, robust=True, cbar_kwargs={'format': '%.1f'})# robust=True,
		axes[0][1].set_title('Longitudinal length scale')
		axes[0][1].set_xlabel('Longitude')
		axes[0][1].set_ylabel('Latitude')

		W_da3.plot(ax=axes[1][0], cmap=cm.plasma, add_colorbar=True, robust=True, cbar_kwargs={'format': '%.1f'})#robust=True,
		axes[1][0].set_title('Rotation')
		axes[1][0].set_xlabel('Longitude')
		axes[1][0].set_ylabel('Latitude')

		W_da4.plot(ax=axes[1][1], cmap=cm.plasma, add_colorbar=True, robust=True, cbar_kwargs={'format': '%.1f'})#robust=True,
		axes[1][1].set_title('Anisotropy ratio')
		axes[1][1].set_xlabel('Longitude')
		axes[1][1].set_ylabel('Latitude')

		plt.tight_layout()
		plt.show()

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
		U, L, Ez, sigma2, loss, loss_tot = utils._cv_spatPCA(
			self.data, 
			phi*self.spatcov_, 
			self.holdout_, 
			k=self.n_components, 
			phi=phi
		)
		
		return (phi, loss_tot)

	def fit(self, lss=[1,3,5,7], phis=[1e3,5e3,1e4,5e4,1e5], rots=[0, 1*np.pi/16, 2*np.pi/16, 3*np.pi/16]):
		#
		if self.nonstationary:
			self._fit_nonstationary(lss, phis, rots)
		else:
			self._fit_stationary(lss, phis, rots)
			
	def _fit_nonstationary(self, lss, phis, rots):
		#
		#fit best ls for fixed init_phi
		#
		ls_combinations = list(itertools.product(lss, lss, rots))
		init_phi = 1e4
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
		#fit best phi for fixed ls
		#		
		results = list(tqdm(
			Parallel(n_jobs=self.n_cores, return_as="generator")(
				delayed(self._evaluate_phi)(phi) 
				for phi in phis
			),
			total = len(phis),
			desc = 'Validating sill'
		))

		# results = Parallel(n_jobs=-1)(
		# 	delayed(self._evaluate_phi)(phi) 
		# 	for phi in tqdm(phis, desc='Validating sill variance for prior spatial covariance.')
		# )

		self.sill_, _ = min(results, key=lambda x: x[1])
		self.spatcov_ = self.sill_ * self.spatcov_

	def _fit_stationary(self, lss, phis, rots):
		#
		#fit best ls for fixed init_phi
		#
		ls_combinations = list(itertools.product(lss, lss))
		init_phi = 1e4
		
		results = list(tqdm(
			Parallel(n_jobs=self.n_cores, return_as='generator')(
				delayed(self._evaluate_ls)(ls1, ls2, init_phi) 
				for ls1, ls2 in ls_combinations
			),
			total = len(ls_combinations),
			desc = 'Validating length scales'
		))
		
		self.ls1_, self.ls2_, _ = min(results, key=lambda x: x[2])
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

	def plot_bivariate_nonstationary_spatcov(self):
		if not self.nonstationary:
			raise ValueError(
				"This plot can only be used when nonstationary=True. "
				"The current model is initialized as stationary."
			)

		if not hasattr(self, 'lam_lat_'):
			raise ValueError("This plot can only be used after .fit() is completed.")

		# --- 1. Extract and format data ---
		def prep_da(data_arr):
			da = xr.DataArray(data=data_arr, dims=('location'), coords={'location': self.data.location})
			return da.unstack().sortby('lon', 'lat')

		W_lat = prep_da(self.lam_lat_)
		W_lon = prep_da(self.lam_lon_)
		W_rot = prep_da(self.rot_)

		lat_vals = W_lat.values
		lon_vals = W_lon.values
		rot_vals = W_rot.values
		
		# Coordinates for plotting
		X = W_lat.lon.values
		Y = W_lat.lat.values

		# --- 2. Bivariate Color Mixing Logic ---
		def normalize(arr):
			"""Min-Max normalization ignoring NaNs"""
			arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
			# Prevent division by zero if array is constant
			if arr_max == arr_min:
				return np.zeros_like(arr)
			return (arr - arr_min) / (arr_max - arr_min)

		def create_bivariate_rgb(var_x, var_y):
			"""
			Uses bilinear interpolation to blend a 4-corner bivariate palette.
			Default Palette: Teal (X) and Pink (Y)
			"""
			norm_x = normalize(var_x)
			norm_y = normalize(var_y)
			
			# Define the 4 corners of the color space (RGB values from 0.0 to 1.0)
			# You can easily swap these numbers to try different palettes!
			c00 = np.array([0.95, 0.93, 0.88])  # Low X, Low Y (Warm Off-White)
			c10 = np.array([0.85, 0.30, 0.40])  # High X, Low Y (Crimson Red)
			c01 = np.array([0.25, 0.45, 0.80])  # Low X, High Y (Cobalt Blue)
			c11 = np.array([0.25, 0.15, 0.35])  # High X, High Y (Dark Violet)
			x = norm_x[..., np.newaxis]
			y = norm_y[..., np.newaxis]
			
			# Bilinear Interpolation Math
			rgb = (1 - x) * (1 - y) * c00 + \
				  x * (1 - y) * c10 + \
				  (1 - x) * y * c01 + \
				  x * y * c11
				  
			# Handle NaNs (Set them to pure white)
			mask = np.isnan(var_x) | np.isnan(var_y)
			rgb[mask] = [1.0, 1.0, 1.0] 
			
			# Ensure values stay strictly between 0 and 1
			return np.clip(rgb, 0, 1)

		# Create the RGB image arrays
		rgb_lat = create_bivariate_rgb(lat_vals, rot_vals)
		rgb_lon = create_bivariate_rgb(lon_vals, rot_vals)

		# --- 3. Plotting ---
		fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 0.4]})

		# Plot 1: Lat & Rot
		axes[0].imshow(rgb_lat, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
		axes[0].set_title('Latitudinal Length Scale & Rotation')
		axes[0].set_xlabel('Longitude')
		axes[0].set_ylabel('Latitude')

		# Plot 2: Lon & Rot
		axes[1].imshow(rgb_lon, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
		axes[1].set_title('Longitudinal Length Scale & Rotation')
		axes[1].set_xlabel('Longitude')
		axes[1].set_ylabel('Latitude')

		# Plot 3: The 2D Bivariate Legend
		ax_leg = axes[2]
		# Create a grid of values from 0 to 1
		x_leg, y_leg = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
		rgb_leg = create_bivariate_rgb(x_leg, y_leg)
		
		ax_leg.imshow(rgb_leg, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
		ax_leg.set_title('Bivariate Legend')
		ax_leg.set_xlabel('Length Scale (Cyan)')
		ax_leg.set_ylabel('Rotation (Magenta)')
		
		# Add interpretive text to the corners of the legend
		ax_leg.text(0.05, 0.05, 'Low / Low', fontsize=9, va='bottom', ha='left')
		ax_leg.text(0.95, 0.05, 'High LS', fontsize=9, va='bottom', ha='right')
		ax_leg.text(0.05, 0.95, 'High Rot', fontsize=9, va='top', ha='left')
		ax_leg.text(0.95, 0.95, 'High Both', fontsize=9, va='top', ha='right', color='white')

		plt.tight_layout()
		plt.show()

	def plot_ellipse_glyphs(self, step=4, scale=1.0):
		if not self.nonstationary:
			raise ValueError("This plot requires a nonstationary model.")

		if not hasattr(self, 'lam_lat_'):
			raise ValueError("Model must be fitted before plotting.")

		# --- 1. Extract and format data ---
		def prep_da(data_arr):
			da = xr.DataArray(data=data_arr, dims=('location'), coords={'location': self.data.location})
			return da.unstack().sortby('lon', 'lat')

		W_lat = prep_da(self.lam_lat_)
		W_lon = prep_da(self.lam_lon_)
		W_rot = prep_da(self.rot_)

		X = W_lat.lon.values
		Y = W_lat.lat.values
		
		lat_vals = W_lat.values
		lon_vals = W_lon.values
		rot_vals = W_rot.values

		# --- 2. Plotting Setup ---
		fig, ax = plt.subplots(figsize=(10, 8))
		
		# Optional: Plot the overall trace (variance) as a faint background color
		# trace = lat_vals**2 + lon_vals**2
		# ax.imshow(trace, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], 
		#           cmap='Greys', alpha=0.3, aspect='auto')

		# --- 3. Draw the Ellipses ---
		# To make the plot readable, we color the ellipses by their anisotropy ratio
		# High ratio (long and skinny) = yellow/red, Low ratio (circular) = dark purple
		ratio = np.maximum(lat_vals, lon_vals) / (np.minimum(lat_vals, lon_vals) + 1e-8)
		norm = plt.Normalize(vmin=np.nanmin(ratio), vmax=np.nanpercentile(ratio, 95))
		cmap = cm.plasma

		for i in range(0, len(Y), step):
			for j in range(0, len(X), step):
				l_lat = lat_vals[i, j]
				l_lon = lon_vals[i, j]
				r = rot_vals[i, j]
				# Skip if data is NaN (e.g., outside the spatial boundary)
				if np.isnan(l_lat) or np.isnan(l_lon) or np.isnan(r):
				    continue
				# width corresponds to x-axis (lon), height to y-axis (lat)
				# We multiply by 2 because length scale represents the semi-axis
				width = 2 * l_lon * scale
				height = 2 * l_lat * scale
				# Matplotlib Ellipse expects rotation in degrees, counter-clockwise
				angle_deg = np.degrees(r)
				color = cmap(norm(ratio[i, j]))
				ellipse = Ellipse(
					xy=(X[j], Y[i]), 
					width=width, 
					height=height, 
					angle=angle_deg,
					edgecolor='black',
					facecolor=color,
					alpha=0.7,
					linewidth=0.5
				)
				ax.add_patch(ellipse)

		# Clean up the plot axes
		ax.set_xlim(X.min(), X.max())
		ax.set_ylim(Y.min(), Y.max())
		ax.set_aspect('equal', adjustable='box') # Keeps ellipses from distorting
		
		ax.set_title(f'Spatially Varying Anisotropy (Subsampled every {step} points)')
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		
		# Add a colorbar for the anisotropy ratio
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label('Anisotropy Ratio (Max LS / Min LS)')

		plt.tight_layout()
		plt.show()

	def plot_vector_glyphs(self, step=4, scale=1.0):
		if not self.nonstationary:
			raise ValueError("This plot requires a nonstationary model.")

		if not hasattr(self, 'lam_lat_'):
			raise ValueError("Model must be fitted before plotting.")

		# --- 1. Extract and format data ---
		def prep_da(data_arr):
			da = xr.DataArray(data=data_arr, dims=('location'), coords={'location': self.data.location})
			return da.unstack().sortby('lon', 'lat')

		W_lat = prep_da(self.lam_lat_)
		W_lon = prep_da(self.lam_lon_)
		W_rot = prep_da(self.rot_)

		X = W_lat.lon.values
		Y = W_lat.lat.values
		
		lat_vals = W_lat.values
		lon_vals = W_lon.values
		rot_vals = W_rot.values

		# --- 2. Subsample via Array Slicing (Much faster than loops) ---
		X_sub = X[::step]
		Y_sub = Y[::step]
		X_grid, Y_grid = np.meshgrid(X_sub, Y_sub)

		lat_sub = lat_vals[::step, ::step]
		lon_sub = lon_vals[::step, ::step]
		rot_sub = rot_vals[::step, ::step]

		# --- 3. Compute Vector Components (U, V) ---
		# Longitudinal (Base axis = X)
		U_lon = lon_sub * np.cos(rot_sub)
		V_lon = lon_sub * np.sin(rot_sub)

		# Latitudinal (Base axis = Y)
		U_lat = lat_sub * (-np.sin(rot_sub))
		V_lat = lat_sub * np.cos(rot_sub)

		# --- 4. Plotting Setup ---
		fig, axes = plt.subplots(1, 2, figsize=(16, 7))
		
		# In ax.quiver, a larger 'scale' parameter actually makes the arrows *smaller*.
		# We invert your 'scale' argument so that higher values = larger arrows visually.
		q_scale = 1.0 / scale if scale > 0 else None

		# Subplot 1: Longitudinal
		# 'pivot=mid' centers the arrow exactly on the grid coordinate
		q1 = axes[0].quiver(
			X_grid, Y_grid, U_lon, V_lon, lon_sub, # Passing lon_sub colors the arrows by magnitude
			pivot='mid', cmap='viridis', scale=q_scale, 
			angles='xy', scale_units='xy', width=0.003
		)
		axes[0].set_title(f'Longitudinal Length Scales & Rotation (Step={step})')
		axes[0].set_xlabel('Longitude')
		axes[0].set_ylabel('Latitude')
		axes[0].set_aspect('equal', adjustable='box')
		fig.colorbar(q1, ax=axes[0], label='Magnitude (Lon Length Scale)')

		# Subplot 2: Latitudinal
		q2 = axes[1].quiver(
			X_grid, Y_grid, U_lat, V_lat, lat_sub, # Passing lat_sub colors the arrows by magnitude
			pivot='mid', cmap='plasma', scale=q_scale, 
			angles='xy', scale_units='xy', width=0.003
		)
		axes[1].set_title(f'Latitudinal Length Scales & Rotation (Step={step})')
		axes[1].set_xlabel('Longitude')
		axes[1].set_ylabel('Latitude')
		axes[1].set_aspect('equal', adjustable='box')
		fig.colorbar(q2, ax=axes[1], label='Magnitude (Lat Length Scale)')

		plt.tight_layout()
		plt.show()



