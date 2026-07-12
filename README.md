# SOFM
Spatially orthogonal factor models
```
import xarray as xr
from sofm import SOFM

# 1. Load your spatial data (ensure it has 'location' indexed with 'lat'/'lon')
# data_xr = ... 

# 2. Initialize the SOFM model for 5 latent factors
sofm_model = SOFM(
    data=data_xr, 
    n_components=5, 
    nonstationary=True, 
    n_cores=8
)

# 3. Fit the model
sofm_model.fit()

# 4. Visualize the extracted spatial domains
fig = sofm_model.plot_loadings(robust=True)
fig.show()

# 5. Access the latent profiles for downstream analysis
latent_factors = sofm_model.Ez_
```
