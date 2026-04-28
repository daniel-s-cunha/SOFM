class SpatialCovariance:
	#Assume data is an xarray data array with "location" as an index for lat lon.

	def __init__(self, data, nonstationary=True,rotation=True,max_lag=20, blk_sz = 10, n_clusters = 80):
		self.data = data
		self.nonstationary = nonstationary
		self.rotation = rotation
		self.max_lag = max_lag
		self.blk_sz = blk_sz
		self.n_clusters = n_clusters
		#
		self.holdout_ = None

	def generate_holdout(self):
		#
		self.holdout_ = _create_mask(self.data,n_clusters=self.n_clusters,blk_sz=self.blk_sz)
		#
		return self
	
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
    
    def fit(
        self,
        lss = [1,3,5,7,9],
        phis = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12]
        ):
        #Set ls values
        #Set phi values
        #Create mask with kmeans
        
        #Fix phi 1e3
        #joblib: run cv_spatpca for all ls combinations
            #---Create Sigma for current params
            #---Run cv_spatPCA
        #choose the best ls
            #---Take nll from across models and choose ls that has smallest nll
        #joblib: run cv_spatpca for all phi values
        #rerun spatPCA with final model #make this optional
        return 
