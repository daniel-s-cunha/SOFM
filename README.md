# SOFM
Spatially orthogonal factor models

## Installation

Please install `scikit-sparse` via conda before installing this package:

```bash
conda install -c conda-forge scikit-sparse
pip install sofm
```

## Data
Please use the `spatialLIBD_to_netcdf.R` script to download the DLPFC data from R and save as a netcdf file to be imported into xarray.

## Code example
```
from sofm import SOFM

# 1. Load netcdf into xarray
# Please see `spatialLIBD_to_netcdf.R` for directions on formatting the netcdf file.
da = xr.open_dataset("/data_directory/spatial_transcriptomic_data.nc", engine='netcdf4')['logcounts']

da = da - da.mean(dim='spot') #SOFM assumes the mean structure has been subtracted
da = da.rename({ #SOFM assumes `da` has `location` dimension indexed by `lat,lon` coordinates
    'spot': 'location', 
    'array_col': 'lon', 
    'array_row': 'lat'
})
da = da.set_index(location=['lat', 'lon'])

# 2. Initialize SOFM model
sofm_model = SOFM(
    data=da, 
    n_components=5, 
    n_cores=-1 #please set number of cores
)

# 3. Fit model
sofm_model.fit()

# 4. Visualize spatial loadings
fig = sofm_model.plot_loadings(robust=True)
fig.show()

# 5. Analyze latent factors
latent_factors = sofm_model.Ez_
```

If helpful, try out the code using a synthetic dataset,
```
import numpy as np
import xarray as xr
from sofm import SOFM

# 1. Generate Synthetic Data
# Create a 30x30 spatial grid
lats, lons = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
lat_flat = lats.flatten()
lon_flat = lons.flatten()
n_locations = len(lat_flat)
n_features = 20

factor1 = np.sin(lat_flat) + np.cos(lon_flat)      
factor2 = np.exp(-(lat_flat**2 + lon_flat**2) / 2) 
factor3 = lat_flat + lon_flat                      
Z = np.column_stack([factor1, factor2, factor3])

np.random.seed(42)
W = np.random.randn(3, n_features) # Synthetic loadings
data_matrix = Z @ W + np.random.randn(n_locations, n_features) * 0.5

da = xr.DataArray(
    data_matrix,
    dims=['location', 'feature'],
    coords={
        'lat': ('location', lat_flat),
        'lon': ('location', lon_flat),
        'feature': np.arange(n_features)
    }
)
da = da.set_index(location=['lat', 'lon'])
da = da - da.mean(dim='location')

# 2. Initialize SOFM model
sofm_model = SOFM(
    data=da, 
    n_components=3, 
    n_cores=-1 #please set number of cores
)

# 3. Fit model
sofm_model.fit()

# 4. Visualize spatial loadings
fig = sofm_model.plot_loadings(robust=True)
fig.show()

# 5. Analyze latent factors
latent_factors = sofm_model.Ez_
```