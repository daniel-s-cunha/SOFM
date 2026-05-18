from sklearn.neighbors import kneighbors_graph
import sympy.printing
import itertools
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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*Sparse CSR tensor support is in beta state.*")
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.sparse_compressed_tensor.*")

def _constr_spat_blk(da,block_sz=10):
    #da should be a single observation of the spatial process; block_sz is the side length
    lats = da.lat.values
    lons = da.lon.values

    unique_lats = np.sort(np.unique(np.round(lats, 6)))
    unique_lons = np.sort(np.unique(np.round(lons, 6)))

    lat_descending = (lats[0] > lats[-1]) if len(lats) > 1 else True

    row_ranks = np.searchsorted(unique_lats, np.round(lats, 6))
    col_ranks = np.searchsorted(unique_lons, np.round(lons, 6))

    if lat_descending:
        row_ranks = (len(unique_lats) - 1) - row_ranks

    block_row = row_ranks // block_sz
    block_col = col_ranks // block_sz

    n_unique_lons = len(unique_lons)
    n_block_cols = (n_unique_lons // block_sz) + 1

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

# def _create_mask(da,n_blocks=80,block_sz=10):
#     #
#     spat_blk = _constr_spat_blk(da, block_sz=block_sz)
#     block_ids = spat_blk.values
#     lats = da.lat.values
#     lons = da.lon.values
#     uniq = np.unique(block_ids)
#     #
#     centers = []
#     valid_uniq = []
#     #
#     for uid in uniq:
#         in_block = (block_ids == uid)
#         if np.any(in_block): 
#             centers.append([np.mean(lats[in_block]), np.mean(lons[in_block])])
#             valid_uniq.append(uid)
            
#     centers = np.array(centers)
#     valid_uniq = np.array(valid_uniq)
#     #
#     kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
#     kmeans.fit(centers)
#     centroids = kmeans.cluster_centers_
#     #
#     tree = cKDTree(centers)
#     _, closest_idx = tree.query(centroids)
#     #
#     mask_ids = valid_uniq[closest_idx]
#     #
#     mask_centers = centers[closest_idx]
#     #
#     mask = torch.tensor(np.where(np.isin(spat_blk, mask_ids), spat_blk, 0),dtype=torch.int16)
#     #
#     return mask, mask_ids, mask_centers

def _create_mask(da, n_blocks=80, block_sz=10):
    #
    spat_blk = _constr_spat_blk(da, block_sz=block_sz)
    
    # Extract to numpy ONCE to prevent Dask from re-evaluating the graph later
    block_ids = spat_blk.values 
    lats = da.lat.values
    lons = da.lon.values
    
    # ---------------------------------------------------------
    # OPTIMIZATION: Replace the O(N*U) loop with a vectorized groupby
    # ---------------------------------------------------------
    df = pd.DataFrame({'block': block_ids, 'lat': lats, 'lon': lons})
    
    # Calculate means for all blocks simultaneously
    grouped = df.groupby('block').mean()
    
    valid_uniq = grouped.index.values
    centers = grouped[['lat', 'lon']].values
    # ---------------------------------------------------------

    #
    kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
    kmeans.fit(centers)
    centroids = kmeans.cluster_centers_
    #
    tree = cKDTree(centers)
    _, closest_idx = tree.query(centroids)
    #
    mask_ids = valid_uniq[closest_idx]
    mask_centers = centers[closest_idx]
    
    # OPTIMIZATION: Pass `block_ids` (numpy array) instead of `spat_blk` (xarray)
    # This prevents triggering a redundant Dask computation.
    mask_array = np.where(np.isin(block_ids, mask_ids), block_ids, 0)
    mask = torch.tensor(mask_array, dtype=torch.int16)
    #
    return mask, mask_ids, mask_centers

def _gen_spat_da(sq_len, k=3, n_samples = 1000, max_lag=30, ls1 = 1, ls2 = 1, nonstationary=True, mode='gradient', sigma=1):
    unique_coords = np.arange(1, sq_len + 1)
    lon_grid, lat_grid = np.meshgrid(unique_coords, unique_coords)
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    total_locations = sq_len ** 2
    random_data = np.random.rand(total_locations)
    da = xr.DataArray(
        data=random_data,
        dims=["location"],
        coords={
            "lat": ("location", lat_flat),
            "lon": ("location", lon_flat)
        }
    )
    phi=1e0
    m = sq_len**2

    if nonstationary:
        lats = da.lat.values
        lons = da.lon.values
        N = len(lats)
        alpha_lat, alpha_lon, t_u, t_v = _gen_synth_spline(
            lats, lons, num_interior_knots=5, mode=mode
        )
        if mode=='circle':
            alpha_rot = np.zeros(alpha_lat.shape[0])
        if mode=='gradient':
            alpha_rot = np.zeros(alpha_lat.shape[0])
        else:
            alpha_rot = alpha_lat/4 #this should yield rotations less than pi/4 ~= np.log(5)/4

        Sigma,lam_lat,lam_lon,rot = _construct_nonstat_cov(
            lats, lons, alpha_lat, alpha_lon, alpha_rot, t_u, t_v, variance=phi, max_lag=max_lag
        )

    else:
        Sigma = _compute_spat_cov_rs(da, phi=phi, length_scale=ls1, length_scale2=ls2, rot=0, max_lag=max_lag)
    #
    _, U = torch.lobpcg(A=Sigma, k=k, largest=True)
    #
    if sigma == -1:
        sigma, L_diag = _optimize_log_prior(U, Sigma, phi, m, k)
    else:
        _, L_diag = _optimize_log_prior(U, Sigma, phi, m, k)
    #
    mt = np.random.uniform(0.75,1.25)
    L_diag = L_diag**mt
    #
    T_new = n_samples
    #
    EZ = np.random.normal(size=(k, T_new))
    #
    sim = U@np.diag(L_diag)@EZ + np.random.normal(loc=0,scale=sigma, size=(m, T_new))
    sim_da = xr.DataArray(
        data=sim,
        dims=('location','gene'),
        coords={
            'location': da.location,
            'lon': da.lon,
            'lat':da.lat
        }
    )
    return sim_da, lam_lat, lam_lon, rot, U, L_diag, sigma

def _optimize_log_prior(U, Sigma, phi, m, k):
    sigma = torch.tensor(1, dtype=torch.float32)
    L_diag = torch.tensor([5*(k - j) for j in range(k)],dtype=torch.float32)
    #
    Ut = U.clone().detach().to(torch.float32)#Ut = torch.tensor(U,dtype=torch.float32)
    SU = Sigma @ Ut
    Ut_Sigma_U = (Ut*SU).sum(dim=0)
    trSigma = torch.tensor(phi*m, dtype = torch.float32)
    #
    log_sigma = torch.nn.Parameter(torch.log(sigma.clone().detach()))
    log_L_diag = torch.nn.Parameter(torch.log(L_diag.clone().detach()))
    optimizer = torch.optim.Adam([log_sigma, log_L_diag], lr=0.05)
    pls = float('inf')
    for i in range(100): # Allow more iterations since M-steps need to converge
        optimizer.zero_grad()
        sigma = torch.exp(log_sigma)
        L_diag = torch.exp(log_L_diag)
        loss = -_log_prior(sigma, L_diag, U, m, Ut_Sigma_U, trSigma, k) 
        ls = loss.item()
        if abs(pls - ls) / (abs(pls) + 1e-8) < 1e-5: 
            break            
        pls = ls
        loss.backward()
        optimizer.step()
    return sigma.detach().numpy(), L_diag.detach().numpy()

def _compute_spat_cov_rs(da,phi=1, length_scale = 1, length_scale2 = -99, rot=0, max_lag=10):
    cov = _construct_index_based_cov(
        da.lat.values, 
        da.lon.values, 
        variance=phi,
        length_scale = length_scale,
        length_scale2 = length_scale2,
        rot = rot,
        max_lag=max_lag
    )
    #
    return _convert_S_tensor(cov)


def _construct_index_based_cov(lats, lons, variance=1.0, length_scale=10.0, length_scale2=-99, rot=0, max_lag=10):
    N = len(lats)
    
    u_lats = np.unique(np.round(lats, 8))
    u_lons = np.unique(np.round(lons, 8))
    
    n_rows = len(u_lats)
    n_cols = len(u_lons)
    
    row_indices = np.searchsorted(u_lats, np.round(lats, 8))
    col_indices = np.searchsorted(u_lons, np.round(lons, 8))
    
    grid_map = np.full((n_rows, n_cols), -1, dtype=np.int32)
    grid_map[row_indices, col_indices] = np.arange(N)

    rows_out = []
    cols_out = []
    data_out = []
    
    rows_out.append(np.arange(N))
    cols_out.append(np.arange(N))
    data_out.append(np.full(N, variance))

    lat_step = np.min(np.diff(u_lats))
    lon_step = np.min(np.diff(u_lons))        
    min_step = min(lat_step, lon_step)
    
    # --- Precompute Stationary Anisotropic Rotation Coefficients ---
    if length_scale2 != -99:
        c = np.cos(rot)
        s = np.sin(rot)
        l1_sq = 2 * length_scale**2
        l2_sq = 2 * length_scale2**2
        
        A = (c**2 / l1_sq) + (s**2 / l2_sq)
        B = (s**2 / l1_sq) + (c**2 / l2_sq)
        C = c * s * (1 / l1_sq - 1 / l2_sq)
    
    for dr in range(-max_lag, max_lag + 1):
        for dc in range(-max_lag, max_lag + 1):
            if dr == 0 and dc == 0: 
                continue 

            if dr >= 0:
                r_src_start, r_src_end = 0, n_rows - dr
                r_dst_start, r_dst_end = dr, n_rows
            else:
                r_src_start, r_src_end = -dr, n_rows
                r_dst_start, r_dst_end = 0, n_rows + dr
                
            if dc >= 0:
                c_src_start, c_src_end = 0, n_cols - dc
                c_dst_start, c_dst_end = dc, n_cols
            else:
                c_src_start, c_src_end = -dc, n_cols
                c_dst_start, c_dst_end = 0, n_cols + dc
            
            # If shift is larger than grid, skip
            if r_src_end <= r_src_start or c_src_end <= c_src_start:
                continue

            src = grid_map[r_src_start:r_src_end, c_src_start:c_src_end].ravel()
            dst = grid_map[r_dst_start:r_dst_end, c_dst_start:c_dst_end].ravel()
            
            mask = (src != -1) & (dst != -1)
            mask &= (src < dst)
            
            if not mask.any():
                continue
                
            u = src[mask]
            v = dst[mask]
            
            dx = lats[u] - lats[v]
            dy = lons[u] - lons[v]
            
            if length_scale2 == -99:
                # Isotropic (rotation has no effect)
                d_sq = (dx**2 + dy**2) / (2 * length_scale**2)
                vals = variance * np.exp(-d_sq)
            else:
                # Anisotropic with rotation
                d_sq = A * dx**2 + B * dy**2 + 2 * C * dx * dy
                vals = variance * np.exp(-d_sq) 
                
            rows_out.append(u)
            cols_out.append(v)
            data_out.append(vals)

    diag_vals = data_out[0]
    
    if len(rows_out) > 1:
        off_rows = np.concatenate(rows_out[1:])
        off_cols = np.concatenate(cols_out[1:])
        off_vals = np.concatenate(data_out[1:])
        
        tri = sp.coo_matrix((off_vals, (off_rows, off_cols)), shape=(N, N))
        
        full_cov = tri + tri.T + sp.diags(diag_vals, format='coo')
    else:
        full_cov = sp.diags(diag_vals, format='coo')
        
    return full_cov.tocsr()

# def _construct_index_based_cov(lats, lons, variance=1.0, length_scale=10.0, length_scale2=-99, rot = 0, max_lag=10):
#     N = len(lats)
    
#     u_lats = np.unique(np.round(lats, 8))
#     u_lons = np.unique(np.round(lons, 8))
    
#     n_rows = len(u_lats)
#     n_cols = len(u_lons)
    
#     row_indices = np.searchsorted(u_lats, np.round(lats, 8))
#     col_indices = np.searchsorted(u_lons, np.round(lons, 8))
    
#     grid_map = np.full((n_rows, n_cols), -1, dtype=np.int32)
#     grid_map[row_indices, col_indices] = np.arange(N)

#     rows_out = []
#     cols_out = []
#     data_out = []
    
#     rows_out.append(np.arange(N))
#     cols_out.append(np.arange(N))
#     data_out.append(np.full(N, variance))

#     lat_step = np.min(np.diff(u_lats))
#     lon_step = np.min(np.diff(u_lons))        
#     min_step = min(lat_step, lon_step)
    
#     for dr in range(-max_lag, max_lag + 1):
#         for dc in range(-max_lag, max_lag + 1):
#             if dr == 0 and dc == 0: 
#                 continue 

#             if dr >= 0:
#                 r_src_start, r_src_end = 0, n_rows - dr
#                 r_dst_start, r_dst_end = dr, n_rows
#             else:
#                 r_src_start, r_src_end = -dr, n_rows
#                 r_dst_start, r_dst_end = 0, n_rows + dr
                
#             if dc >= 0:
#                 c_src_start, c_src_end = 0, n_cols - dc
#                 c_dst_start, c_dst_end = dc, n_cols
#             else:
#                 c_src_start, c_src_end = -dc, n_cols
#                 c_dst_start, c_dst_end = 0, n_cols + dc
            
#             # If shift is larger than grid, skip
#             if r_src_end <= r_src_start or c_src_end <= c_src_start:
#                 continue

#             src = grid_map[r_src_start:r_src_end, c_src_start:c_src_end].ravel()
#             dst = grid_map[r_dst_start:r_dst_end, c_dst_start:c_dst_end].ravel()
            
#             mask = (src != -1) & (dst != -1)
            
#             mask &= (src < dst)
            
#             if not mask.any():
#                 continue
                
#             u = src[mask]
#             v = dst[mask]
#             if length_scale2==-99:
#                 #isotropic
#                 d_sq = (lats[u] - lats[v])**2 + (lons[u] - lons[v])**2                
#                 vals = variance * np.exp(-d_sq / (2*length_scale**2))
#             else:
#                 #anisotropic
#                 d_sq = (lats[u] - lats[v])**2 / (2*length_scale**2) + (lons[u] - lons[v])**2 / (2*length_scale2**2)
#                 vals = variance * np.exp(-d_sq) 
#             rows_out.append(u)
#             cols_out.append(v)
#             data_out.append(vals)

#     diag_vals = data_out[0]
    
#     if len(rows_out) > 1:
#         off_rows = np.concatenate(rows_out[1:])
#         off_cols = np.concatenate(cols_out[1:])
#         off_vals = np.concatenate(data_out[1:])
        
#         tri = sp.coo_matrix((off_vals, (off_rows, off_cols)), shape=(N, N))
        
#         full_cov = tri + tri.T + sp.diags(diag_vals, format='coo')
#     else:
#         full_cov = sp.diags(diag_vals, format='coo')
        
#     return full_cov.tocsr()

def _convert_S_tensor(scp_matrix):
    data = scp_matrix.data
    indices = scp_matrix.indices
    indptr = scp_matrix.indptr
    
    t_data = torch.as_tensor(data,dtype=torch.float32)
    t_indices = torch.from_numpy(indices).to(torch.int32)
    t_indptr = torch.from_numpy(indptr).to(torch.int32)
    
    torch_csr = torch.sparse_csr_tensor(
        t_indptr, 
        t_indices, 
        t_data, 
        size=scp_matrix.shape
    )    
    return torch_csr

def _log_prior(sigma, L_diag, U, m, Ut_Sigma_U, trSigma,k):
    #
    #
    sigma2 = sigma**2
    L2 = L_diag**2
    lambda_vals = 1.0 / (L2 + sigma2) # (L^2 + s^2)^-1
    # t1 = -T * (m / 2.0) * torch.log(sigma2)
    inner_diag = -L2 / (sigma2 * (L2 + sigma2)) #lambda_vals - (1/sigma2)
    tr_part1 = torch.sum(inner_diag * Ut_Sigma_U)
    #FIXME: change tr_part2 to be 0 in the singular case
    tr_part2 = (1/sigma2) * trSigma 
    t3 = -0.5 * (tr_part1 + tr_part2)
    #
    lambda_sorted, _ = torch.sort(lambda_vals, descending=False)
    diff_matrix = lambda_sorted.unsqueeze(0) - lambda_sorted.unsqueeze(1)
    rows, cols = torch.triu_indices(k, k, offset=1)
    t4 = torch.sum(torch.log(diff_matrix[rows, cols]))
    #
    #t5 = torch.sum(torch.log((1/sigma2) - lambda_vals)) #
    t5 = (m - k) * torch.sum(torch.log((1/sigma2) - lambda_vals))
    #
    const_6 = (m - k)*(m - k - 1)/2.0 + 3.0#(n - m)*(m - k) +
    t6 = -0.5 * const_6 * torch.log(sigma2)
    #
    const_7 = 3.0#(n - m) +
    t7 = -0.5 * const_7 * torch.sum(torch.log(L2 + sigma2))
    #
    t8 = torch.sum(torch.log(torch.abs(L_diag)))
    #
    terms = {
        't3': t3, 't4': t4, 
        't5': t5, 't6': t6, 't7': t7, 't8': t8
    }
    #print(terms)
    #total = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
    total = sum(terms.values())
    #
    return total/m

def _spatPCA(
    Y_da,
    Sigma,
    k: int,
    phi = 1,
    lr_s = 1e-1,
    lr_l = 1e-2,
    df = 1,
    init_sigma = 1,
    max_em_iter: int = 20,
    tol_em: float = 1e-5,
    verb = True
):
    #Setup
    #
    m = Y_da.shape[0]
    T = Y_da.shape[1]
    n = m+df
    Y = torch.tensor(Y_da.values, dtype=torch.float32)
    term1 = (Y**2).sum()
    trSigma = torch.tensor(phi*m, dtype = torch.float32)
    #
    # 3) initialize
    np.random.seed(42)
    ULVT = randomized_svd(Y_da.values, n_components=k,n_iter=1)
    U = torch.tensor(ULVT[0], dtype=torch.float32)
    SU = Sigma@U
    Lams = ULVT[1]**2/T
    sigma2 = (term1.numpy()/T - np.sum(Lams))/(m-k)
    L_approx = (Lams - sigma2)**0.5
    L_diag = torch.tensor(L_approx, dtype=torch.float32, requires_grad=True) 
    if sigma2<=0:
        sigma2 = 0.01
    sigma = torch.tensor(sigma2**0.5, dtype=torch.float32, requires_grad=True)
    #
    prev_ll = -torch.inf
    #
    sigma2 = sigma**2
    L_tilde = -(1/(L_diag**2 + sigma2) - 1/sigma2) #torch.diag
    #SU = Sigma @ U 
    #
    log_sigma = torch.nn.Parameter(torch.log(sigma.clone().detach()))
    log_L_diag = torch.nn.Parameter(torch.log(L_diag.clone().detach()))
    optimizer = torch.optim.Adam([log_sigma, log_L_diag], lr=lr_s)
    #
    for iteration in range(max_em_iter):        
        #
        # E-step
        with torch.no_grad():
            #
            M = (L_diag**2 + sigma2)     #W.T @ W  +  sigma2 * torch.eye(k,dtype=torch.float32)
            M_inv = torch.diag(1/M) #torch.linalg.inv(M)
            #
            Ez = M_inv @ (U * L_diag.unsqueeze(0)).T @ Y
            Ezz = T * sigma2 * M_inv + Ez @ Ez.T 
            sum_yz = Y @ Ez.T 
            sum_zz = Ezz
            #
            #
            for _ in range(5):
                U_old = U
                G = sum_yz * L_diag.unsqueeze(0) / sigma2  + SU * L_tilde.unsqueeze(0)
                ULVT = torch.linalg.svd(G, full_matrices=False)
                U = ULVT[0] @ ULVT[2]
                SU = Sigma @ U
                if (abs(U - U_old)).sum() <1e-1:
                    break
            Ut_Sigma_U = (U*SU).sum(dim=0)
        #
        #
        #
        pls = float('inf')
        for i in range(100): # Allow more iterations since M-steps need to converge
            optimizer.zero_grad()
            sigma = torch.exp(log_sigma)
            L_diag = torch.exp(log_L_diag)
            loss = _neg_lik_samp(sigma, L_diag, U, Ez, Ezz, m, T, Y, term1, Ut_Sigma_U, trSigma, n, k) 
            ls = loss.item()
            if abs(pls - ls) / (abs(pls) + 1e-8) < 1e-5: 
                break
                
            pls = ls
            loss.backward()
            optimizer.step()
        sigma2 = sigma**2
        L_tilde = -(1/(L_diag**2 + sigma2) - 1/sigma2)#torch.diag
        #
        if (abs(ls - prev_ll) < tol_em * abs(prev_ll)) and (iteration > 5):
            break
        prev_ll = ls        
    #
    #
    L_diag, indices = torch.sort(L_diag, descending=True)
    U = U[:, indices]
    Ez = Ez[indices, :]
    return U, L_diag, Ez, sigma2,ls

def _cv_spatPCA(
    Y_da,
    Sigma,
    mask,
    k: int,
    phi = 1,
    lr_s = 1e-1,
    lr_l = 1e-2,
    df = 1,
    sigma2_init_set = [1],
    max_em_iter: int = 6,
    tol_em: float = 1e-3,
    verb = False
):
    mask_ids = mask #these will be used for single block MSEs
    mask = (mask != 0) #this is used to hold out data
    #
    m = Y_da.shape[0] #total locations including those held out
    T = Y_da.shape[1]
    n = m+df
    m0 = m - mask.sum() #total observed locations inlcuding held-out
    #
    perm = torch.argsort(mask.long())
    reverse_perm = torch.argsort(perm)
    #
    mask_ids = mask_ids[perm]
    #
    Y_temp = torch.as_tensor(Y_da.values.copy(), dtype=torch.float32, device=perm.device)[perm]
    Y = Y_temp[0:m0,:]
    Yp = Y_temp[m0:,:] #only to be used for validation, not in training
    term1 = (Y**2).sum()
    trSigma = torch.tensor(phi*m, dtype = torch.float32)
    #
    # Initialize
    np.random.seed(42)
    Y_aug = torch.cat((Y, torch.tensor(np.random.normal(0,1,(m-m0,Y.shape[1])),dtype=torch.float32)),dim=0) #(Sigma/phi) @
    ULVT = randomized_svd(Y_aug.numpy(), n_components=k,n_iter=1)
    U = torch.tensor(ULVT[0], dtype=torch.float32)
    U_rp = U[reverse_perm,:]
    SU_rp = Sigma@U_rp
    Lams = ULVT[1]**2/T
    sigma2 = (term1.numpy()/T - np.sum(Lams))/(m-k)
    L_approx = (Lams - sigma2)**0.5
    L_diag = torch.tensor(L_approx, dtype=torch.float32, requires_grad=True) 
    #print('INITIAL L: ',L_diag)
    if sigma2<=0:
        sigma2 = 0.01
    sigma = torch.tensor(sigma2**0.5, dtype=torch.float32, requires_grad=True)
    #
    prev_ll = -torch.inf
    #
    sigma2 = sigma**2
    L_tilde = -(1/(L_diag**2 + sigma2) - 1/sigma2)
    #SU = Sigma @ U
    SU = (Sigma@U_rp)[perm,:]
    #
    log_sigma = torch.nn.Parameter(torch.log(sigma.clone().detach()))
    log_L_diag = torch.nn.Parameter(torch.log(L_diag.clone().detach()))
    optimizer = torch.optim.Adam([log_sigma, log_L_diag], lr=lr_s)
    #
    for iteration in range(max_em_iter):        
        #
        # E-step
        with torch.no_grad():
            #print("starting E-step...")
            #
            Uo = U[0:m0,:]
            M = (L_diag.unsqueeze(1)*Uo.T@Uo*L_diag.unsqueeze(0) + sigma2*torch.eye(k))
            ch = torch.linalg.cholesky(M)
            M_inv = torch.cholesky_solve(torch.eye(k),ch)
            #
            Ez = M_inv @ (Uo * L_diag.unsqueeze(0)).T @ Y #Only used observed U's
            Ezz = T * sigma2 * M_inv + Ez @ Ez.T 
            sum_yz = Y @ Ez.T #(m0 x T) @ (T x k) = (m0 x k)
            Upt = U[m0:,:].detach().clone()
            term12 = _comp_trEyp_yp(Y,U,L_diag,sigma2,M_inv,m,m0,T)
            #Ezyt = Ezz * L_diag.unsqueeze(0) @ Up_old.T #Do this in loss function to save memory
            #print("completed E-step...")
            L_diag_t = L_diag.detach().clone()
            #
            #print("starting M-step U...")
            for jj in range(5):
                U_old = U
                G = torch.cat((sum_yz, (Upt*L_diag.unsqueeze(0))@Ezz),dim=0) * L_diag.unsqueeze(0) / sigma2 + SU * L_tilde.unsqueeze(0)
                ULVT = torch.linalg.svd(G, full_matrices=False)
                U = ULVT[0] @ ULVT[2]
                #SU = Sigma @ U
                SU = (Sigma@U[reverse_perm,:])[perm,:]
                if (abs(U - U_old)).sum() <1e-1:
                    #print("M-step U finished iteration: ",jj)
                    break
            
            Ut_Sigma_U = (U*SU).sum(dim=0)
            loss = _neg_lik_val(sigma, L_diag, U, L_diag_t, Ez, Ezz, m, T, Y, term1, term12, Upt, Ut_Sigma_U, trSigma, n, k, m0) #compute_negative_likelihood(sigma, L_diag)
            ls = loss.item()
            #print(f"Q after U M-step: {-ls:.5f}",f" Iteration: {iteration}")
        #
        pls = float('inf')
        for i in range(100): # Allow more iterations since M-steps need to converge
            optimizer.zero_grad()
            sigma = torch.exp(log_sigma)
            L_diag = torch.exp(log_L_diag)
            loss = _neg_lik_val(sigma, L_diag, U, L_diag_t, Ez, Ezz, m, T, Y, term1, term12, Upt, Ut_Sigma_U, trSigma, n, k, m0)
            ls = loss.item()
            if abs(pls - ls) / (abs(pls) + 1e-8) < 1e-5: 
                #if verb: 
                    #print(f"Prior loss {pls:.5f} and current loss {ls:.5f}")
                    #print(f"L_diag: {L_diag.detach().numpy()}")
                    #print(f"sigma: {sigma.detach().numpy():.5f}")
                    #print(f"M-Step for L and sigma converged at iter {i}")
                break
                
            pls = ls
            loss.backward()
            optimizer.step()

        sigma2 = sigma**2
        L_tilde = -(1/(L_diag**2 + sigma2) - 1/sigma2)
        #
        #print(f"Q: {-ls:.5f}",f" Iteration: {iteration}")
        if (abs(ls - prev_ll) < tol_em * abs(prev_ll)) and (iteration > 2):
            break
        prev_ll = ls        
    #
    #
    nlls,nll_tot = _comp_nll(Yp,Y,U,L_diag,sigma2,M_inv,m,m0,mask_ids)
    L_diag, indices = torch.sort(L_diag, descending=True)
    U = U[:, indices]
    Ez = Ez[indices, :]
    return U[reverse_perm,:], L_diag, Ez, sigma2, nlls,nll_tot

def _comp_trEyp_yp(Y,U,L_diag,sigma2,M_inv,m,m0,T):
    #This calculates the sum_i trEyipyip'
    Uo = U[0:m0,:]
    Up = U[m0:,:]
    L = L_diag.unsqueeze(0)
    #
    UoTUo = Uo.T@Uo
    UpTUp = Up.T @ Up    
    #
    t1 = sigma2 * torch.trace(UpTUp * L @ M_inv * L) 
    t2 = sigma2*(m-m0)
    #    
    Eyp = (Up * L) @ M_inv @ (Uo * L).T @ Y
    t3 = torch.sum(Eyp**2)
    #
    trEyp_yp = T*(t1 + t2) + t3
    #
    return trEyp_yp

def _neg_lik_samp(sigma, L_diag, U, Ez, Ezz, m, T, Y, term1, Ut_Sigma_U, trSigma, n,k):
    #
    #
    #sorted_L, indices = torch.sort(L_diag, descending=True)
    #U_sorted = U[:, indices]
    #Ez_sorted = Ez[indices, :]
    #Ezz_sorted = Ezz[indices, :][:, indices]
    #
    #
    sigma2 = sigma**2
    L2 = L_diag**2
    lambda_vals = 1.0 / (L2 + sigma2) # (L^2 + s^2)^-1
    t1 = -T * (m / 2.0) * torch.log(sigma2)
    #
    UL = U * L_diag.unsqueeze(0)
    trace_A = 2.0 * torch.sum((Y.t() @ UL) * Ez.t())
    Ezz_diag = torch.diagonal(Ezz, dim1=-2, dim2=-1)
    trace_B = torch.sum(Ezz_diag * L2.unsqueeze(0))
    t2 = -0.5 * (1/sigma2) * (term1 - trace_A + trace_B)
    #
    inner_diag = -L2 / (sigma2 * (L2 + sigma2)) #lambda_vals - (1/sigma2)
    tr_part1 = torch.sum(inner_diag * Ut_Sigma_U)
    #FIXME: change tr_part2 to be 0 in the singular case
    tr_part2 = (1/sigma2) * trSigma 
    t3 = -0.5 * (tr_part1 + tr_part2)
    #
    lambda_sorted, _ = torch.sort(lambda_vals, descending=False)
    diff_matrix = lambda_sorted.unsqueeze(0) - lambda_sorted.unsqueeze(1)
    rows, cols = torch.triu_indices(k, k, offset=1)
    t4 = torch.sum(torch.log(diff_matrix[rows, cols]))
    #
    #FIXME: singularWishart update
    #t5 = torch.sum(torch.log((1/sigma2) - lambda_vals)) #
    t5 = (m - k) * torch.sum(torch.log((1/sigma2) - lambda_vals))
    #
    #FIXME: singularWishart update
    const_6 = (m - k)*(m - k - 1)/2.0 + 3.0#(n - m)*(m - k) +
    t6 = -0.5 * const_6 * torch.log(sigma2)
    #
    const_7 = 3.0#(n - m) +
    t7 = -0.5 * const_7 * torch.sum(torch.log(L2 + sigma2))
    #
    t8 = torch.sum(torch.log(torch.abs(L_diag)))
    #
    terms = {
        't1': t1, 't2': t2, 't3': t3, 't4': t4, 
        't5': t5, 't6': t6, 't7': t7, 't8': t8
    }
    #total = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
    total = sum(terms.values())
    #
    return -total/(T*m)

def _neg_lik_val(sigma, L_diag, U, L_diag_t, Ez, Ezz, m, T, Y, term1, term12, Upt, Ut_Sigma_U, trSigma, n, k, m0):
    #
    sigma2 = sigma**2
    L2 = L_diag**2
    lambda_vals = 1.0 / (L2 + sigma2) # (L^2 + s^2)^-1
    t1 = -T * (m / 2.0) * torch.log(sigma2)
    #
    Up = U[m0:,:]
    UoL = U[0:m0,:] * L_diag.unsqueeze(0)
    #
    trace_A = 2.0 * torch.sum((Y.t() @ UoL) * Ez.t())
    L_mat = torch.diag(L_diag); L_mat_t = torch.diag(L_diag_t)
    trace_A2 = 2.0*torch.trace(Ezz @ L_mat_t @ Upt.T @ Up @ L_mat) #2.0 * torch.trace((L_mat @ Ezz @ L_mat) @ (Upt.T @ Upt))
    Ezz_diag = torch.diagonal(Ezz, dim1=-2, dim2=-1)
    trace_B = torch.sum(Ezz_diag * L2.unsqueeze(0))
    t2 = -0.5 * (1/sigma2) * (term1 + term12 - trace_A - trace_A2 + trace_B)
    #
    inner_diag = -L2 / (sigma2 * (L2 + sigma2)) #lambda_vals - (1/sigma2)
    tr_part1 = torch.sum(inner_diag * Ut_Sigma_U)
    tr_part2 = (1/sigma2) * trSigma #torch.trace(Sigma)
    t3 = -0.5 * (tr_part1 + tr_part2)
    #
    lambda_sorted, _ = torch.sort(lambda_vals, descending=False)
    diff_matrix = lambda_sorted.unsqueeze(0) - lambda_sorted.unsqueeze(1)
    rows, cols = torch.triu_indices(k, k, offset=1)
    t4 = torch.sum(torch.log(diff_matrix[rows, cols]))
    #
    #FIXME: singularWishart update
    t5 = (m - k) * torch.sum(torch.log((1/sigma2) - lambda_vals))
    #
    #FIXME: singularWishart update
    const_6 = (m - k)*(m - k - 1)/2.0 + 3.0 #(n - m)*(m - k) +
    t6 = -0.5 * const_6 * torch.log(sigma2)
    #
    #FIXME: singularWishart update
    const_7 = 3.0 #(n - m) +
    t7 = -0.5 * const_7 * torch.sum(torch.log(L2 + sigma2))
    #
    t8 = torch.sum(torch.log(torch.abs(L_diag)))
    #
    total = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
    return -total/(T*m)

def _comp_nll(Yp,Y,U,L_diag,sigma2,M_inv,m,m0,mask_ids):
    T = Y.shape[1]
    mask_ids = mask_ids[m0:] #mask_ids should already be permuted with observed first, masked second
    #
    Uo = U[0:m0,:]
    Up = U[m0:,:]
    L = L_diag.unsqueeze(0)
    #
    Eyp = (Up * L) @ M_inv @ (Uo * L).T @ Y
    # nll = torch.mean((Yp - Eyp)**2)
    #
    nll = ((Yp - Eyp)**2).sum(dim=1).detach().numpy()/T
    mask_ids = mask_ids.numpy()
    #
    nll_df = pd.DataFrame({'mask_id':mask_ids, 'nll':nll})
    nlls = nll_df.groupby('mask_id')['nll'].mean()
    #
    nll_tot = ((Yp - Eyp)**2).mean()
    #
    return nlls,nll_tot

def _fit_spline(da, data_array, gamma_grid=np.logspace(-1, 3, 15), knot_grid=[10], degree=3):
    ls_lat = data_array[:, 0]
    ls_lon = data_array[:, 1]
    rot_hat = data_array[:, 2]
    lats = data_array[:, 3]
    lons = data_array[:, 4]
    domain_lats = da.lat.values
    domain_lons = da.lon.values
    
    y_lat = np.log(ls_lat)
    y_lon = np.log(ls_lon)
    y_rot = rot_hat
    
    N = len(lats)
    
    pad_lat = (np.max(domain_lats) - np.min(domain_lats)) * 0.01
    pad_lon = (np.max(domain_lons) - np.min(domain_lons)) * 0.01
    
    domain_lat_min, domain_lat_max = np.min(domain_lats) - pad_lat, np.max(domain_lats) + pad_lat
    domain_lon_min, domain_lon_max = np.min(domain_lons) - pad_lon, np.max(domain_lons) + pad_lon

    best_gcv_lat = np.inf
    best_gcv_lon = np.inf
    best_gcv_rot = np.inf
    best_params = None
    best_alpha_lat = None
    best_alpha_lon = None
    best_alpha_rot = None
    best_t_u = None
    best_t_v = None

    # Outer loop: Number of knots
    for num_knots in knot_grid:
        u_knots = np.linspace(domain_lat_min, domain_lat_max, num_knots)
        t_u = np.r_[[u_knots[0]] * degree, u_knots, [u_knots[-1]] * degree]
        
        v_knots = np.linspace(domain_lon_min, domain_lon_max, num_knots)
        t_v = np.r_[[v_knots[0]] * degree, v_knots, [v_knots[-1]] * degree]
        
        B_u = BSpline.design_matrix(lats, t_u, degree).toarray()
        B_v = BSpline.design_matrix(lons, t_v, degree).toarray()
        
        K_u = B_u.shape[1]
        K_v = B_v.shape[1]
        K_total = K_u * K_v
        
        B = np.einsum('ik,il->ikl', B_v, B_u).reshape(N, K_total)
        
        D_u = np.diff(np.eye(K_u), n=2, axis=0)
        D_v = np.diff(np.eye(K_v), n=2, axis=0)
        
        Du_T_Du = D_u.T @ D_u
        Dv_T_Dv = D_v.T @ D_v
        
        P_u = np.kron(np.eye(K_v), Du_T_Du)
        P_v = np.kron(Dv_T_Dv, np.eye(K_u))
        
        BtB = B.T @ B
        rhs_lat = B.T @ y_lat
        rhs_lon = B.T @ y_lon
        rhs_rot = B.T @ y_rot

        # Inner loop: Smoothing parameters
        for g_u, g_v in itertools.product(gamma_grid, gamma_grid):
            lhs = BtB + g_u * P_u + g_v * P_v
            
            try:
                # FIX 2: Use linear solve instead of explicit inversion.
                # Solves LHS * H = BtB --> H = LHS^-1 * BtB
                # The trace of H is the exact Effective Degrees of Freedom
                H_core = np.linalg.solve(lhs, BtB)
                edf = np.trace(H_core)
                
                # Get the alpha coefficients
                alpha_lat = np.linalg.solve(lhs, rhs_lat)
                alpha_lon = np.linalg.solve(lhs, rhs_lon)
                alpha_rot = np.linalg.solve(lhs, rhs_rot)
            except np.linalg.LinAlgError:
                continue 
            
            y_hat_lat = B @ alpha_lat
            y_hat_lon = B @ alpha_lon
            y_hat_rot = B @ alpha_rot
            
            sse_lat = np.sum((y_lat - y_hat_lat)**2)
            sse_lon = np.sum((y_lon - y_hat_lon)**2)
            sse_rot = np.sum((y_rot - y_hat_rot)**2)
            
            # FIX 3: GCV Inflation factor to aggressively penalize undersmoothing
            inflation_factor = 1#1.4
            effective_N_penalty = inflation_factor * edf
            
            # Guard against the squared denominator flipping negative values to positive
            if effective_N_penalty >= N:
                continue
                
            denom = (1 - effective_N_penalty / N)**2
            
            gcv_lat = (sse_lat / N) / denom
            gcv_lon = (sse_lon / N) / denom
            gcv_rot = (sse_rot / N) / denom
            
            total_gcv = gcv_lat + gcv_lon + gcv_rot
            
            if  gcv_lat < best_gcv_lat:
                best_gcv_lat = gcv_lat
                best_params_lat = (g_u, g_v, num_knots)
                best_alpha_lat = alpha_lat
            if  gcv_lon < best_gcv_lon:
                best_gcv_lon = gcv_lon
                best_params_lon = (g_u, g_v, num_knots)
                best_alpha_lon = alpha_lon
            if  gcv_rot < best_gcv_rot:
                best_gcv_rot = gcv_rot
                best_params_rot = (g_u, g_v, num_knots)
                best_alpha_rot = alpha_rot

    print(f"Latitudinal spline | Knots: {best_params_lat[2]} | gamma_u: {best_params_lat[0]:.2e} | gamma_v: {best_params_lat[1]:.2e}")
    print(f"Longitudinal spline | Knots: {best_params_lon[2]} | gamma_u: {best_params_lon[0]:.2e} | gamma_v: {best_params_lon[1]:.2e}")
    print(f"Rotational spline | Knots: {best_params_rot[2]} | gamma_u: {best_params_rot[0]:.2e} | gamma_v: {best_params_rot[1]:.2e}")

    return best_alpha_lat, best_alpha_lon, best_alpha_rot, t_u, t_v

# def _fit_spline(da, data_array, gamma_u=5, gamma_v=5, num_interior_knots=10, gamma_grid=np.logspace(-1, 5, 15), knot_grid=[5], degree=3):
#     ls_lat = data_array[:, 0]
#     ls_lon = data_array[:, 1]
#     lats = data_array[:, 2]
#     lons = data_array[:, 3]
#     domain_lats = da.lat.values
#     domain_lons = da.lon.values
    
#     y_lat = np.log(ls_lat)
#     y_lon = np.log(ls_lon)
    
#     N = len(lats)
    
#     pad_lat = (np.max(domain_lats) - np.min(domain_lats)) * 0.01
#     pad_lon = (np.max(domain_lons) - np.min(domain_lons)) * 0.01
    
#     domain_lat_min, domain_lat_max = np.min(domain_lats) - pad_lat, np.max(domain_lats) + pad_lat
#     domain_lon_min, domain_lon_max = np.min(domain_lons) - pad_lon, np.max(domain_lons) + pad_lon

#     best_gcv = np.inf
#     best_params = None
#     best_alpha_lat = None
#     best_alpha_lon = None
#     best_t_u = None
#     best_t_v = None

#     # Outer loop: Number of knots (Expensive basis creation)
#     for num_knots in knot_grid:
#         u_knots = np.linspace(domain_lat_min, domain_lat_max, num_knots)
#         t_u = np.r_[[u_knots[0]] * degree, u_knots, [u_knots[-1]] * degree]
        
#         v_knots = np.linspace(domain_lon_min, domain_lon_max, num_knots)
#         t_v = np.r_[[v_knots[0]] * degree, v_knots, [v_knots[-1]] * degree]
        
#         B_u = BSpline.design_matrix(lats, t_u, degree).toarray()
#         B_v = BSpline.design_matrix(lons, t_v, degree).toarray()
        
#         K_u = B_u.shape[1]
#         K_v = B_v.shape[1]
#         K_total = K_u * K_v
        
#         B = np.einsum('ik,il->ikl', B_v, B_u).reshape(N, K_total)
        
#         D_u = np.diff(np.eye(K_u), n=2, axis=0)
#         D_v = np.diff(np.eye(K_v), n=2, axis=0)
        
#         Du_T_Du = D_u.T @ D_u
#         Dv_T_Dv = D_v.T @ D_v
        
#         P_u = np.kron(np.eye(K_v), Du_T_Du)
#         P_v = np.kron(Dv_T_Dv, np.eye(K_u))
        
#         BtB = B.T @ B
#         rhs_lat = B.T @ y_lat
#         rhs_lon = B.T @ y_lon

#         # Inner loop: Smoothing parameters (Fast matrix inversion)
#         for g_u, g_v in itertools.product(gamma_grid, gamma_grid):
#             lhs = BtB + g_u * P_u + g_v * P_v
            
#             try:
#                 # Invert LHS once per gamma combination
#                 lhs_inv = np.linalg.inv(lhs)
#             except np.linalg.LinAlgError:
#                 continue # Skip if singular
            
#             alpha_lat = lhs_inv @ rhs_lat
#             alpha_lon = lhs_inv @ rhs_lon
            
#             y_hat_lat = B @ alpha_lat
#             y_hat_lon = B @ alpha_lon
            
#             sse_lat = np.sum((y_lat - y_hat_lat)**2)
#             sse_lon = np.sum((y_lon - y_hat_lon)**2)
            
#             # Fast calculation of trace(lhs_inv @ BtB) using element-wise product
#             edf = np.sum(lhs_inv * BtB.T)
            
#             denom = (1 - edf / N)**2
            
#             if denom <= 0:
#                 continue
                
#             gcv_lat = (sse_lat / N) / denom
#             gcv_lon = (sse_lon / N) / denom
            
#             # Minimize the combined GCV of both latitudinal and longitudinal length scales
#             total_gcv = gcv_lat + gcv_lon
            
#             if total_gcv < best_gcv:
#                 best_gcv = total_gcv
#                 best_params = (g_u, g_v, num_knots)
#                 best_alpha_lat = alpha_lat
#                 best_alpha_lon = alpha_lon
#                 best_t_u = t_u
#                 best_t_v = t_v

#     print(f"Optimal Spline GCV: {best_gcv:.4f} | ", f"Knots: {best_params[2]} | ", f"gamma_u: {best_params[0]:.2e} | ", f"gamma_v: {best_params[1]:.2e}")

#     return best_alpha_lat, best_alpha_lon, best_t_u, best_t_v

# def _fit_spline(da, data_array, gamma_u=5, gamma_v=5, num_interior_knots=10, degree=3):
#     #
#     ls_lat = data_array[:, 0]
#     ls_lon = data_array[:, 1]
#     lats = data_array[:, 2]
#     lons = data_array[:, 3]
#     domain_lats = da.lat.values
#     domain_lons = da.lon.values
#     #
#     y_lat = np.log(ls_lat)
#     y_lon = np.log(ls_lon)
    
#     N = len(lats)
    
#     pad_lat = (np.max(domain_lats) - np.min(domain_lats)) * 0.01
#     pad_lon = (np.max(domain_lons) - np.min(domain_lons)) * 0.01
    
#     domain_lat_min, domain_lat_max = np.min(domain_lats) - pad_lat, np.max(domain_lats) + pad_lat
#     domain_lon_min, domain_lon_max = np.min(domain_lons) - pad_lon, np.max(domain_lons) + pad_lon

#     u_knots = np.linspace(domain_lat_min, domain_lat_max, num_interior_knots)
#     t_u = np.r_[[u_knots[0]] * degree, u_knots, [u_knots[-1]] * degree]
    
#     v_knots = np.linspace(domain_lon_min, domain_lon_max, num_interior_knots)
#     t_v = np.r_[[v_knots[0]] * degree, v_knots, [v_knots[-1]] * degree]
    
#     B_u = BSpline.design_matrix(lats, t_u, degree).toarray()
#     B_v = BSpline.design_matrix(lons, t_v, degree).toarray()
    
#     K_u = B_u.shape[1]
#     K_v = B_v.shape[1]
#     K_total = K_u * K_v
    
#     B = np.einsum('ik,il->ikl', B_v, B_u).reshape(N, K_total)
    
#     D_u = np.diff(np.eye(K_u), n=2, axis=0)
#     D_v = np.diff(np.eye(K_v), n=2, axis=0)
    
#     Du_T_Du = D_u.T @ D_u
#     Dv_T_Dv = D_v.T @ D_v
    
#     P_u = np.kron(np.eye(K_v), Du_T_Du)
#     P_v = np.kron(Dv_T_Dv, np.eye(K_u))
    
#     BtB = B.T @ B
#     lhs = BtB + gamma_u * P_u + gamma_v * P_v
    
#     rhs_lat = B.T @ y_lat
#     rhs_lon = B.T @ y_lon
    
#     alpha_lat = np.linalg.solve(lhs, rhs_lat)
#     alpha_lon = np.linalg.solve(lhs, rhs_lon)
    
#     return alpha_lat, alpha_lon, t_u, t_v

# def _construct_nonstat_cov(lats, lons, alpha_lat, alpha_lon, t_u, t_v, variance=1.0, max_lag=10, degree=3):
#     N = len(lats)
        
#     B_u = BSpline.design_matrix(lats, t_u, degree).toarray()
#     B_v = BSpline.design_matrix(lons, t_v, degree).toarray()

#     B = np.einsum('ik,il->ikl', B_v, B_u).reshape(N, -1)
#     lam_lat = np.exp(B @ alpha_lat)
#     lam_lon = np.exp(B @ alpha_lon)
    
#     # --- 2. Grid Mapping for Fast Neighbor Search ---
#     u_lats = np.unique(np.round(lats, 8))
#     u_lons = np.unique(np.round(lons, 8))
    
#     n_rows = len(u_lats)
#     n_cols = len(u_lons)
    
#     row_indices = np.searchsorted(u_lats, np.round(lats, 8))
#     col_indices = np.searchsorted(u_lons, np.round(lons, 8))
    
#     grid_map = np.full((n_rows, n_cols), -1, dtype=np.int32)
#     grid_map[row_indices, col_indices] = np.arange(N)

#     rows_out = [np.arange(N)]
#     cols_out = [np.arange(N)]
#     data_out = [np.full(N, variance)]
    
#     for dr in range(-max_lag, max_lag + 1):
#         for dc in range(-max_lag, max_lag + 1):
#             if dr == 0 and dc == 0: 
#                 continue 

#             if dr >= 0:
#                 r_src_start, r_src_end = 0, n_rows - dr
#                 r_dst_start, r_dst_end = dr, n_rows
#             else:
#                 r_src_start, r_src_end = -dr, n_rows
#                 r_dst_start, r_dst_end = 0, n_rows + dr
                
#             if dc >= 0:
#                 c_src_start, c_src_end = 0, n_cols - dc
#                 c_dst_start, c_dst_end = dc, n_cols
#             else:
#                 c_src_start, c_src_end = -dc, n_cols
#                 c_dst_start, c_dst_end = 0, n_cols + dc
            
#             if r_src_end <= r_src_start or c_src_end <= c_src_start:
#                 continue

#             src = grid_map[r_src_start:r_src_end, c_src_start:c_src_end].ravel()
#             dst = grid_map[r_dst_start:r_dst_end, c_dst_start:c_dst_end].ravel()
            
#             mask = (src != -1) & (dst != -1)
#             mask &= (src < dst)
            
#             if not mask.any():
#                 continue
                
#             u = src[mask]
#             v = dst[mask]
            
#             lam_lat_u, lam_lat_v = lam_lat[u], lam_lat[v]
#             lam_lon_u, lam_lon_v = lam_lon[u], lam_lon[v]
            
#             avg_var_lat = (lam_lat_u**2 + lam_lat_v**2) / 2.0
#             avg_var_lon = (lam_lon_u**2 + lam_lon_v**2) / 2.0
            
#             det_u = lam_lat_u * lam_lon_u
#             det_v = lam_lat_v * lam_lon_v
#             det_avg = np.sqrt(avg_var_lat * avg_var_lon)
            
#             S_uv = np.sqrt(det_u * det_v) / det_avg
            
#             Q_uv = ((lats[u] - lats[v])**2 / avg_var_lat) + ((lons[u] - lons[v])**2 / avg_var_lon)
            
#             vals = variance * S_uv * np.exp(-0.5 * Q_uv)
#             #vals = variance * S_uv * (1 - Q_uv) * np.exp(-0.5 * Q_uv)

#             rows_out.append(u)
#             cols_out.append(v)
#             data_out.append(vals)

#     diag_vals = data_out[0]
    
#     if len(rows_out) > 1:
#         off_rows = np.concatenate(rows_out[1:])
#         off_cols = np.concatenate(cols_out[1:])
#         off_vals = np.concatenate(data_out[1:])
        
#         tri = sp.coo_matrix((off_vals, (off_rows, off_cols)), shape=(N, N))
        
#         full_cov = tri + tri.T + sp.diags(diag_vals, format='coo')
#     else:
#         full_cov = sp.diags(diag_vals, format='coo')
        
#     return _convert_S_tensor(full_cov.tocsr()),lam_lat,lam_lon

def _construct_nonstat_cov(lats, lons, alpha_lat, alpha_lon, alpha_rot, t_u, t_v, variance=1.0, max_lag=10, degree=3):
    N = len(lats)
        
    B_u = BSpline.design_matrix(lats, t_u, degree).toarray()
    B_v = BSpline.design_matrix(lons, t_v, degree).toarray()

    B = np.einsum('ik,il->ikl', B_v, B_u).reshape(N, -1)
    lam_lat = np.exp(B @ alpha_lat)
    lam_lon = np.exp(B @ alpha_lon)
    rot = B @ alpha_rot
    
    lam_lat_sq = lam_lat**2
    lam_lon_sq = lam_lon**2
    cos_rot = np.cos(rot)
    sin_rot = np.sin(rot)
    
    # Elements of the local 2x2 covariance matrices for each location
    a_cov = lam_lat_sq * cos_rot**2 + lam_lon_sq * sin_rot**2
    b_cov = lam_lat_sq * sin_rot**2 + lam_lon_sq * cos_rot**2
    c_cov = (lam_lat_sq - lam_lon_sq) * cos_rot * sin_rot
    
    # --- 2. Grid Mapping for Fast Neighbor Search ---
    u_lats = np.unique(np.round(lats, 8))
    u_lons = np.unique(np.round(lons, 8))
    
    n_rows = len(u_lats)
    n_cols = len(u_lons)
    
    row_indices = np.searchsorted(u_lats, np.round(lats, 8))
    col_indices = np.searchsorted(u_lons, np.round(lons, 8))
    
    grid_map = np.full((n_rows, n_cols), -1, dtype=np.int32)
    grid_map[row_indices, col_indices] = np.arange(N)

    rows_out = [np.arange(N)]
    cols_out = [np.arange(N)]
    data_out = [np.full(N, variance)]
    
    for dr in range(-max_lag, max_lag + 1):
        for dc in range(-max_lag, max_lag + 1):
            if dr == 0 and dc == 0: 
                continue 

            if dr >= 0:
                r_src_start, r_src_end = 0, n_rows - dr
                r_dst_start, r_dst_end = dr, n_rows
            else:
                r_src_start, r_src_end = -dr, n_rows
                r_dst_start, r_dst_end = 0, n_rows + dr
                
            if dc >= 0:
                c_src_start, c_src_end = 0, n_cols - dc
                c_dst_start, c_dst_end = dc, n_cols
            else:
                c_src_start, c_src_end = -dc, n_cols
                c_dst_start, c_dst_end = 0, n_cols + dc
            
            if r_src_end <= r_src_start or c_src_end <= c_src_start:
                continue

            src = grid_map[r_src_start:r_src_end, c_src_start:c_src_end].ravel()
            dst = grid_map[r_dst_start:r_dst_end, c_dst_start:c_dst_end].ravel()
            
            mask = (src != -1) & (dst != -1)
            mask &= (src < dst)
            
            if not mask.any():
                continue
                
            u = src[mask]
            v = dst[mask]
            
            # Distance vectors
            dx = lats[u] - lats[v]
            dy = lons[u] - lons[v]
            
            # Average covariance matrix elements: Sigma_avg = (Sigma_u + Sigma_v) / 2
            a_avg = (a_cov[u] + a_cov[v]) / 2.0
            b_avg = (b_cov[u] + b_cov[v]) / 2.0
            c_avg = (c_cov[u] + c_cov[v]) / 2.0
            
            # Determinant of the averaged covariance matrix
            det_avg = a_avg * b_avg - c_avg**2
            
            # The determinants of the individual matrices are invariant to rotation
            det_u = lam_lat[u] * lam_lon[u]  # This is sqrt(|Sigma_u|)
            det_v = lam_lat[v] * lam_lon[v]  # This is sqrt(|Sigma_v|)
            
            # S_uv scaling factor
            S_uv = np.sqrt(det_u * det_v) / np.sqrt(det_avg)
            
            # Explicit 2x2 matrix inverse for the quadratic form: (X^T * Sigma_avg^-1 * X)
            Q_uv = (b_avg * dx**2 - 2 * c_avg * dx * dy + a_avg * dy**2) / det_avg
            
            vals = variance * S_uv * np.exp(-0.5 * Q_uv)

            rows_out.append(u)
            cols_out.append(v)
            data_out.append(vals)

    diag_vals = data_out[0]
    
    if len(rows_out) > 1:
        off_rows = np.concatenate(rows_out[1:])
        off_cols = np.concatenate(cols_out[1:])
        off_vals = np.concatenate(data_out[1:])
        
        tri = sp.coo_matrix((off_vals, (off_rows, off_cols)), shape=(N, N))
        
        full_cov = tri + tri.T + sp.diags(diag_vals, format='coo')
    else:
        full_cov = sp.diags(diag_vals, format='coo')
        
    return _convert_S_tensor(full_cov.tocsr()), lam_lat, lam_lon, rot

# def _gen_synth_spline(lats, lons, num_interior_knots=5, degree=3, mode='gradient'):
#     """
#     Generates synthetic spline parameters (alpha_lat, alpha_lon, t_u, t_v)
#     to feed into the nonstationary covariance constructor.
#     """
#     # 1. Define boundaries based on the input data (with slight padding)
#     pad_lat = (np.max(lats) - np.min(lats)) * 0.01
#     pad_lon = (np.max(lons) - np.min(lons)) * 0.01
    
#     domain_lat_min, domain_lat_max = np.min(lats) - pad_lat, np.max(lats) + pad_lat
#     domain_lon_min, domain_lon_max = np.min(lons) - pad_lon, np.max(lons) + pad_lon
    
#     # 2. Construct knot vectors (t_u for latitude, t_v for longitude)
#     u_knots = np.linspace(domain_lat_min, domain_lat_max, num_interior_knots)
#     t_u = np.r_[[u_knots[0]] * degree, u_knots, [u_knots[-1]] * degree]
    
#     v_knots = np.linspace(domain_lon_min, domain_lon_max, num_interior_knots)
#     t_v = np.r_[[v_knots[0]] * degree, v_knots, [v_knots[-1]] * degree]
    
#     # Calculate the number of basis functions created by these knots
#     K_u = len(t_u) - degree - 1
#     K_v = len(t_v) - degree - 1
#     K_total = K_v * K_u  # Order matters here to match your einsum!
    
#     # 3. Generate the alpha coefficients
#     # Recall that in your constructor: lam = exp(B @ alpha)
#     # Therefore, alpha is in log-space. An alpha of 0 -> length scale of 1.0. 
#     # An alpha of 1.6 -> length scale of ~5.0.
    
#     if mode == 'gradient':
#         # Creates a smooth gradient where length scales get larger 
#         # as you move North and East.
        
#         # Create a grid of values from 0.5 to 2.0 (length scales ~1.6 to ~7.3)
#         # We shape it (K_v, K_u) so that flattening it perfectly matches
#         # the (N, K_v, K_u) reshape in your einsum logic.
#         val_v, val_u = np.meshgrid(
#             np.linspace(0., 1.609, K_v), 
#             np.linspace(0., 1.609, K_u), 
#             indexing='ij'
#         )
        
#         # Latitudinal length scales change based on latitude (u)
#         alpha_lat = val_u.flatten() 
#         # Longitudinal length scales change based on longitude (v)
#         alpha_lon = val_v.flatten() 
        
#     elif mode == 'random':
#         # Creates a smooth but randomized nonstationary field
#         #np.random.seed(42)
#         # Centered around 1.0 (length scale ~2.7) with some variance
#         alpha_lat = np.random.normal(loc=1.0, scale=0.4, size=K_total)
#         alpha_lon = np.random.normal(loc=1.0, scale=0.4, size=K_total)
        
#     elif mode == 'constant':
#         # Effectively turns your nonstationary model into a stationary one
#         # for baseline testing. (Length scale = exp(1.5) = 4.48 everywhere)
#         alpha_lat = np.full(K_total, 1.5)
#         alpha_lon = np.full(K_total, 1.5)
        
#     else:
#         raise ValueError("mode must be 'gradient', 'random', or 'constant'")
        
#     return alpha_lat, alpha_lon, t_u, t_v

import numpy as np

def _gen_synth_spline(lats, lons, num_interior_knots=5, degree=3, mode='gradient'):
    """
    Generates synthetic spline parameters (alpha_lat, alpha_lon, t_u, t_v)
    to feed into the nonstationary covariance constructor.
    """
    # 1. Define boundaries based on the input data (with slight padding)
    pad_lat = (np.max(lats) - np.min(lats)) * 0.01
    pad_lon = (np.max(lons) - np.min(lons)) * 0.01
    
    domain_lat_min, domain_lat_max = np.min(lats) - pad_lat, np.max(lats) + pad_lat
    domain_lon_min, domain_lon_max = np.min(lons) - pad_lon, np.max(lons) + pad_lon
    
    # 2. Construct knot vectors (t_u for latitude, t_v for longitude)
    u_knots = np.linspace(domain_lat_min, domain_lat_max, num_interior_knots)
    t_u = np.r_[[u_knots[0]] * degree, u_knots, [u_knots[-1]] * degree]
    
    v_knots = np.linspace(domain_lon_min, domain_lon_max, num_interior_knots)
    t_v = np.r_[[v_knots[0]] * degree, v_knots, [v_knots[-1]] * degree]
    
    # Calculate the number of basis functions created by these knots
    K_u = len(t_u) - degree - 1
    K_v = len(t_v) - degree - 1
    K_total = K_v * K_u  # Order matters here to match your einsum!
    
    # 3. Generate the alpha coefficients
    # Recall that in your constructor: lam = exp(B @ alpha)
    # Therefore, alpha is in log-space. An alpha of 0 -> length scale of 1.0. 
    # An alpha of 1.6 -> length scale of ~5.0.
    
    if mode == 'gradient':
        # Creates a smooth gradient where length scales get larger 
        # as you move North and East.
        
        # Create a grid of values from 0.5 to 2.0 (length scales ~1.6 to ~7.3)
        # We shape it (K_v, K_u) so that flattening it perfectly matches
        # the (N, K_v, K_u) reshape in your einsum logic.
        val_v, val_u = np.meshgrid(
            np.linspace(0., 1.609, K_v), 
            np.linspace(0., 1.609, K_u), 
            indexing='ij'
        )
        
        # Latitudinal length scales change based on latitude (u)
        alpha_lat = val_u.flatten() 
        # Longitudinal length scales change based on longitude (v)
        alpha_lon = val_v.flatten() 
        
    elif mode == 'circle':
        # Creates a spatial domain where length scales are long inside a centered
        # circle (occupying ~50% of the area) and short outside of it.
        
        # Approximate control point spatial locations across the domain
        cp_lons, cp_lats = np.meshgrid(
            np.linspace(domain_lon_min, domain_lon_max, K_v),
            np.linspace(domain_lat_min, domain_lat_max, K_u),
            indexing='ij'
        )
        
        # Find the center of the domain
        center_lon = (domain_lon_min + domain_lon_max) / 2.0
        center_lat = (domain_lat_min + domain_lat_max) / 2.0
        
        # Normalize the distances so the domain goes from -1 to 1 in both directions.
        # This allows us to map a perfect circle in normalized space to an ellipse 
        # that scales naturally with the actual map dimensions.
        w = domain_lon_max - domain_lon_min
        h = domain_lat_max - domain_lat_min
        
        norm_lon = (cp_lons - center_lon) / (w / 2.0)
        norm_lat = (cp_lats - center_lat) / (h / 2.0)
        
        # Squared distance from the center in normalized space
        sq_dist = norm_lon**2 + norm_lat**2
        
        # The area of the normalized domain is 2 * 2 = 4.
        # We want the circle to take up half the area (Area = 2).
        # pi * r^2 = 2  =>  r^2 = 2 / pi
        r_sq_threshold = 1.0 / np.pi
        
        # Set alphas: ~5.0 inside the circle, ~1.0 outside the circle
        long_scale = 1.609  # exp(1.609) ≈ 5.0
        short_scale = 0.0   # exp(0) = 1.0
        
        # Apply the boolean mask
        alpha_grid = np.where(sq_dist <= r_sq_threshold, long_scale, short_scale)
        
        alpha_lat = alpha_grid.flatten()
        alpha_lon = alpha_grid.flatten()
        
    elif mode == 'constant':
        # Effectively turns your nonstationary model into a stationary one
        # for baseline testing. (Length scale = exp(1.5) = 4.48 everywhere)
        alpha_lat = np.full(K_total, 1.5)
        alpha_lon = np.full(K_total, 1.5)
        
    else:
        raise ValueError("mode must be 'gradient', 'circle', or 'constant'")
        
    return alpha_lat, alpha_lon, t_u, t_v

def _construct_omega_matrix(da, k_pos=12, k_neg=50, lambda_neg=1.0, random_state=2):
    """
    Constructs the contrastive graph matrix Omega = A^+ - lambda * A^-
    for Spectral Contrastive Loss optimization.
    
    Parameters:
    -----------
    da : xr.DataArray
        The spatial DataArray with a 'location' MultiIndex (lat, lon).
    k_pos : int
        Number of spatial neighbors to define as positive pairs. 
        (Typically 6 for Visium hexagonal grids, 4 for square grids).
    k_neg : int
        Number of randomly sampled distant spots to define as negative pairs per spot.
    lambda_neg : float
        The penalty weight for negative pairs.
        
    Returns:
    --------
    Omega_csr : scipy.sparse.csr_matrix
        The final contrastive matrix.
    A_pos_csr : scipy.sparse.csr_matrix
        The positive adjacency matrix.
    A_neg_csr : scipy.sparse.csr_matrix
        The negative adjacency matrix.
    """
    # 1. Extract coordinates from the standardized xarray MultiIndex
    lats = da.coords['lat'].values
    lons = da.coords['lon'].values
    coords = np.column_stack((lats, lons))
    N = len(coords)
    
    # 2. Construct the Positive Adjacency Matrix (A+)
    # We use KNN to find direct spatial neighbors. include_self=False ensures 
    # we don't try to contrast a spot with itself.
    A_pos = kneighbors_graph(coords, n_neighbors=k_pos, mode='connectivity', include_self=False)
    
    # Make A+ symmetric (if i is a neighbor of j, j is a neighbor of i)
    A_pos = A_pos.maximum(A_pos.T).tocsr()
    
    # 3. Construct the Negative Adjacency Matrix (A-) via Negative Sampling
    np.random.seed(random_state)
    
    # Repeat each spot index k_neg times
    row_indices = np.repeat(np.arange(N), k_neg)
    # Randomly select k_neg distant targets for each spot
    col_indices = np.random.randint(0, N, size=N * k_neg)
    
    # Filter out self-loops immediately
    valid_mask = (row_indices != col_indices)
    row_indices = row_indices[valid_mask]
    col_indices = col_indices[valid_mask]
    
    # Build the initial sparse negative matrix
    data_neg = np.ones(len(row_indices))
    A_neg = sp.coo_matrix((data_neg, (row_indices, col_indices)), shape=(N, N)).tocsr()
    
    # Make A- symmetric 
    A_neg = A_neg.maximum(A_neg.T)
    
    # 4. Clean Overlaps
    # Ensure no negative edges accidentally overlap with positive edges.
    # We do this quickly by element-wise multiplying A_neg by A_pos and subtracting the intersection.
    overlap = A_neg.multiply(A_pos)
    A_neg = A_neg - overlap
    
    # 5. Construct Omega = A+ - lambda * A-
    Omega = A_pos - lambda_neg * A_neg
    
    return _convert_S_tensor(Omega.tocsr())
