# SOFM
Spatially orthogonal factor models
Please use the `spatialLIBD_to_netcdf.R` script to download the DLPFC data from R and save as a netcdf file to be imported into xarray.

```
import xarray as xr
from sofm import SOFM

# 1. Load netcdf into xarray
da = xr.open_dataset("/projectnb/modislc/users/danc/dlpfc_netcdf_data/151675_spatial_expression.nc", engine='netcdf4')['logcounts']

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
