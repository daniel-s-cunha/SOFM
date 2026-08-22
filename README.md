# SOFM
Spatially orthogonal factor models

## Installation

Because SOFM relies on sparse matrix optimizations, you must install `scikit-sparse` via conda before installing this package:

```bash
conda install -c conda-forge scikit-sparse
pip install sofm
```

## Data
Please use the `spatialLIBD_to_netcdf.R` script to download the DLPFC data from R and save as a netcdf file to be imported into xarray.

```
import xarray as xr
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
