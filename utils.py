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

def _constr_spat_blk(da,blk_sz=10):
    #da should be a single observation of the spatial process; blk_sz is the number of pixels/spots per block
    lats = da.lat.values
    lons = da.lon.values

    unique_lats = np.sort(np.unique(np.round(lats, 6)))
    unique_lons = np.sort(np.unique(np.round(lons, 6)))

    lat_descending = (lats[0] > lats[-1]) if len(lats) > 1 else True

    row_ranks = np.searchsorted(unique_lats, np.round(lats, 6))
    col_ranks = np.searchsorted(unique_lons, np.round(lons, 6))

    if lat_descending:
        row_ranks = (len(unique_lats) - 1) - row_ranks

    block_row = row_ranks // blk_sz
    block_col = col_ranks // blk_sz

    n_unique_lons = len(unique_lons)
    n_block_cols = (n_unique_lons // blk_sz) + 1

    block_ids = block_row * n_block_cols + block_col
    #
    spat_dims = da.lat.dims
    spat_coords = {k: v for k, v in da.coords.items() if set(v.dims).issubset(spat_dims)}
    #
    final_mask = xr.DataArray(
        block_ids,
        coords=spat_coords,
        dims=spat_dims
    )
    return(final_mask)

def _create_mask(da,n_clusters=80,blk_sz=10):
    #
    spat_blk = _constr_spat_blk(ds.mean(dim='time'), blk_sz=blk_sz)
    block_ids = spat_blk.values
    lats = ds.lat.values
    lons = ds.lon.values
    uniq = np.unique(block_ids)

    # 2. Compute the true geographic (lat, lon) center of every valid block
    centers = []
    valid_uniq = []

    for uid in uniq:
        in_block = (block_ids == uid)
        if np.any(in_block): # Ensure we only look at blocks with actual land pixels
            centers.append([np.mean(lats[in_block]), np.mean(lons[in_block])])
            valid_uniq.append(uid)
            
    centers = np.array(centers)
    valid_uniq = np.array(valid_uniq)
    #
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(centers)
    centroids = kmeans.cluster_centers_
    #
    tree = cKDTree(centers)
    _, closest_idx = tree.query(centroids)
    #
    evenly_spaced_block_ids = valid_uniq[closest_idx]
    #
    mask = torch.tensor(np.where(np.isin(spat_blk, evenly_spaced_block_ids), spat_blk, 0),dtype=torch.int16)
    #
    return mask






















