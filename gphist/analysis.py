"""Analysis and plotting functions.
"""

import numpy as np
import scipy.interpolate

def get_quantiles(d,q,weights=None,num_bins=100):
	nd,nz = d.shape
	quantiles = np.empty((len(q),nz))
	for iz in range(nz):
		# Histogram the distance values at this redshift. Use the density option
		# to ensure that the bin contents are floats.
		hist,edges = np.histogram(d[:,iz],bins=num_bins,density=True,weights=weights)
		# Build the CDF from this histogram.
		cdf = np.empty_like(edges)
		cdf[0] = 0.
		np.cumsum(hist,out=cdf[1:])
		cdf /= cdf[-1]
		# Linearly interpolate the inverse CDF at the specified quantiles. Remove any
		# (almost) empty bins from the interpolation data to ensure that the x values
		# are (sufficiently) increasing.
		use = np.empty(len(cdf),dtype=bool)
		use[0] = True
		use[1:] = np.diff(cdf) > 1e-8
		inv_cdf = scipy.interpolate.InterpolatedUnivariateSpline(cdf[use],edges[use],k=1)
		quantiles[:,iz] = inv_cdf(q)
	return quantiles
