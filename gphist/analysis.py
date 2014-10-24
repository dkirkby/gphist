"""Analysis and plotting functions.
"""

import numpy as np
import scipy.interpolate

def calculate_posteriors_nll(DH,DA,posteriors):
	"""Calculate -logL for each combination of posterior and prior sample.

	Args:
		DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
		DA(ndarray): Array of shape (nsamples,nz-1) of DA(z) values to use.
		posteriors(list): List of posteriors to use. Each posterior must
			implement a method get_nll(DH,DA) and return an array of nsamples
			-logL values.

	Returns:
		ndarray: An array of shape (npost,nsamples) containing the nll values
			calculated for each posterior independently.
	"""
	nsamples = len(DH)
	npost = len(posteriors)
	nll = np.empty((npost,nsamples))
	for ipost,post in enumerate(posteriors):
		nll[ipost] = post.get_nll(DH,DA)
	return nll

def histogram(data,num_bins,min_value,max_value,weights=None,out=None):
	"""Build a histogram with underflow and overflow bins.

	Uses the numpy.histogram function but also accumulates underflow and
	overflow values. Use the :py:func:`quantiles` function to extract quantile
	levels from the resulting histograms. Use fixed histogram binning to
	simplify combining histograms from different runs.

	Args:
		data(ndarray): Array of data to histogram.
		num_bins(int): Number of equally spaced bins to use for each histogram.
		min_value(float): Minimum data value corresponding to the left edge of
			the first bin. Values below this are accumulated in an underflow bin.
		max_value(float): Maximum data value corresponding to the right edge of
			the last bin. Values above this are accumulated in an overflow bin.
		weights(ndarray): Optional array of weights to use. If provided, its
			length must match the length of data.
		out(ndarray): Optional array where results should be saved. If this is not
			provided, new array memory is allocated.

	Returns:
		ndarray: Array of length num_bins+2 containing per-bin sums of
			weights in each bin. The first and last bin contents
			record underflow and overflow sums of weights, respectively.
			A newly allocated array always has dtype of numpy.float64, independently
			of whether weights are provided. Returns the input parameter out
			when this is not None.

	Raises:
		ValueError: the 'out' array provided does not have the expected shape
			(num_bins+2,).
	"""
	if out is None:
		out = np.empty((num_bins+2,),dtype=np.float64)
	elif out.shape != (num_bins+2,):
		raise ValueError('Arg out must have shape (%d,)' % numbins+2)
	contents,edges = np.histogram(data,bins=num_bins,
		range=(min_value,max_value),weights=weights)
	# Contents will have integral type if there are no weights.
	out[1:-1] = contents.astype(np.float64,copy=False)
	# Calculate underflow and overflow.
	under = data < min_value
	over = data >= max_value
	if weights is None:
		out[0] = np.count_nonzero(under)
		out[-1] = np.count_nonzero(over)
	else:
		out[0] = np.sum(weights[under])
		out[-1] = np.sum(weights[over])
	return out

def calculate_distance_histograms(DH,DH0,DA,DA0,nll,num_bins = 200,min_value=0.,max_value=2.):
	"""Build histograms of DH/DH0 and DA/DA0.

	Calculate histograms for all permutations of posterior weightings.
	"""
	nsamples,nz = DH.shape
	npost = len(nll)
	# Check sizes.
	assert DH0.shape == (nz,)
	assert DA0.shape == (nz-1,)
	assert DA.shape == (nsamples,nz-1)
	assert nll.shape == (npost,nsamples)
	# Allocate output arrays.
	nperm = 2**npost
	DH_hist = np.empty((nperm,nz,num_bins+2))
	DA_hist = np.empty((nperm,nz-1,num_bins+2))
	# Loop over permutations.
	bits = 2**np.arange(npost)
	for iperm in range(nperm):
		# Calculate weights for this permutation.
		mask = np.bitwise_and(iperm,bits) > 0
		perm_nll = np.sum(nll[mask],axis=0)  # Returns zero when mask entries are all False.
		perm_weights = np.exp(-perm_nll)
		# Build nz histograms of DH/DH0.
		for iz in range(nz):
			histogram(DH[:,iz]/DH0[iz],num_bins,min_value,max_value,
				out=DH_hist[iperm,iz],weights=perm_weights)
		# Build nz-1 histograms of DA/DA0.
		for iz in range(nz-1):
			histogram(DA[:,iz]/DA0[iz],num_bins,min_value,max_value,
				out=DA_hist[iperm,iz],weights=perm_weights)
	return DH_hist,DA_hist

def get_histograms(data,weights=None,num_bins=200,min_value=0.,max_value=2.):
	"""Build weighted histograms of each column in an array.

	Uses the numpy.histogram function but also accumulates underflow and
	overflow values. Use the :py:func:`get_quantiles` function to extract quantile
	levels from the resulting histograms. We use fixed histogram binning to
	simplify combining histograms from different runs.

	Args:
		data(ndarray): Array of shape (nrows,ncols) whose columns will
			be histogrammed.
		weights(ndarray): Optional array of weights to use with shape (ncols,).
		num_bins(int): Number of equally spaced bins to use for each histogram.
		min_value(float): Minimum data value corresponding to the left edge of
			the first bin. Values below this are accumulated in an underflow bin.
		max_value(float): Maximum data value corresponding to the right edge of
			the last bin. Values above this are accumulated in an overflow bin.

	Returns:
		ndarray: Array of shape (ncols,num_bins+2) containing per-bin sums of
			weights for column i in row i. The first and last bin contents
			record underflow and overflow sums of weights, respectively.
	"""
	nrows,ncols = data.shape
	histograms = np.empty((ncols,num_bins+2),dtype=np.float64)
	for icol in range(ncols):
		# Calculate the histogram bin contents, which will be integer if there are no weights.
		hist,edges = np.histogram(data[:,icol],bins=num_bins,
			range=(min_value,max_value),weights=weights)
		# Convert contents to float and offset to make room for underflow bin.
		histograms[icol,1:-1] = hist.astype(np.float64,copy=False)
	return histograms

def get_quantiles(d,q,weights=None,num_bins=100):
	"""...
	"""
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
