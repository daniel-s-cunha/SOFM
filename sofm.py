import utils

class SOFM:
    #Assume data is an xarray data array with "location" as an index for lat lon.
    def __init__(self, n_factors, nonstationary = True, rotation = True, prebuilt = False):
        self.n_factors = n_factors
        self.nonstationary = nonstationary
        self.rotation = rotation
        self.prebuilt = prebuilt
        #
        self.is_fitted_ = False
        self.loadings_ = None
        self.factors_ = None
        self.noise_ = None
        self.explained_var_ = None


