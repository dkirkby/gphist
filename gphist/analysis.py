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

def histogram(data,num_bins,bin_range,weights=None,out=None):
	"""Build a histogram with underflow and overflow bins.

	Uses the numpy.histogram function but also accumulates underflow and
	overflow values. Use the :py:func:`quantiles` function to extract quantile
	levels from the resulting histograms. Use fixed histogram binning to
	simplify combining histograms from different runs.

	Args:
		data(ndarray): Array of data to histogram.
		num_bins(int): Number of equally spaced bins to use for each histogram.
		bin_range(ndarray): Array of length 2 with binning min,max values to use.
			Values below or above these limits are accumulated in an underflow
			or overflow bin.
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
		ValueError: the bin_range or out arguments do not have the expected shape.
	"""
	min_value,max_value = bin_range
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

def get_permutations(n):
	"""Builds an array of permutations.

	Args:
		n(int): Length of the array to permute.

	Returns:
		ndarray: Array of booleans with shape (2*n,n). Element (i,j) is True if
			element j belongs to the i-th permutation.
	"""
	nperm = 2**n
	bits = 2**np.arange(n)
	mask = np.empty((nperm,n),dtype=bool)
	for iperm in range(nperm):
		mask[iperm] = np.bitwise_and(iperm,bits) > 0
	return mask

def calculate_distance_histograms(DH,DH0,DA,DA0,nll,num_bins,bin_range):
	"""Build histograms of DH/DH0 and DA/DA0.

	Calculate histograms for all permutations of posterior weightings.

	Args:
		DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
		DH0(ndarray): Array of shape (nz,) used to normalize each DH(z).
		DA(ndarray): Array of shape (nsamples,nz-1) of DA(z) values to use.
		DA0(ndarray): Array of shape (nz-1,) used to normalize each DA(z).
		nll(ndarray): Array of shape (npost,nsamples) containing the nll
			posterior weights to use.
		num_bins(int): Number of equally spaced bins to use for each histogram.
		bin_range(ndarray): Array of length 2 with binning min,max values to use.
			Values below or above these limits are accumulated in an underflow
			or overflow bin.

	Returns:
		tuple: Arrays of histograms for DH/DH0 and DA/DA0 with shapes
			(nperm,nz,num_bins+2) and (nperm,nz-1,num_bins+2), respectively,
			where nperm = 2**npost. The mapping between permutations and the
			permutation index is given by the binary representation of the index.
			For example, iperm = 5 = 2^0 + 2^2 combines posteriors 0 and 2.

	Raises:
		AssertionError: Unexpected sizes of DH,DH0,DA,DA0.
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
	perms = get_permutations(npost)
	for iperm,perm in enumerate(perms):
		# Calculate weights for this permutation.
		perm_nll = np.sum(nll[perm],axis=0)  # Returns zero when perm entries are all False.
		perm_weights = np.exp(-perm_nll)
		# Build nz histograms of DH/DH0.
		for iz in range(nz):
			histogram(DH[:,iz]/DH0[iz],num_bins,bin_range,
				out=DH_hist[iperm,iz],weights=perm_weights)
		# Build nz-1 histograms of DA/DA0.
		for iz in range(nz-1):
			histogram(DA[:,iz]/DA0[iz],num_bins,bin_range,
				out=DA_hist[iperm,iz],weights=perm_weights)
	return DH_hist,DA_hist

def quantiles(histogram,quantile_levels,bin_range,threshold=1e-8):
	"""Calculate quantiles of a histogram.

	The quantiles are estimated by inverse linear interpolation of the cummulative
	histogram. It is probably possible to vectorize this algorithm over histograms
	if there is a bottleneck here.

	Args:
		histogram(ndarray): Array of nbins+2 histogram contents, with bins [0] and [-1]
			containing under- and overflow contents, respectively.
		quantile_levels(ndarray): Array of quantile levels to calculate. Must be in the
			range (0,1) but do not need to be sorted.
		bin_range(ndarray): Array of length 2 containing the [min,max) range corresponding
			to the histogram contents in bins [1:-1].
		threshold(float): Any histogram bins whose normalized contents fall below this
			threshold will be ignored, to minimize loss of precision in the interpolation.

	Raises:
		ValueError: bin_range does not have 2 elements.
		RuntimeError: Some levels fall outside of bin_range so cannot be calculated.
	"""
	# Reconstruct the histogram binning.
	num_bins = len(histogram)-2
	min_value,max_value = bin_range # Raises ValueError if there are not exactly 2 values to unpack.
	bin_edges = np.linspace(min_value,max_value,num_bins+1,endpoint=True)
	# Build the cummulative distribution function, including the under/overflow bins.
	cdf = np.cumsum(histogram)
	cdf /= cdf[-1]
	# Check that the requested levels lie within the binning range.
	if np.min(quantile_levels) < cdf[0]:
		raise RuntimeError('Quantile level %f is below binning range' % np.min(levels))
	if np.max(quantile_levels) > cdf[-2]:
		raise RuntimeError('Quantile level %f is above binning range' % np.max(levels))
	# Skip almost empty bins so that CDF values are increasing for inverse interpolation.
	use = np.diff(cdf) > threshold
	use[0] = True
	# Linearly interpolate CDF levels to estimate the corresponding bin values.
	inv_cdf = scipy.interpolate.InterpolatedUnivariateSpline(
		cdf[use[:-1]],bin_edges[use[:-1]],k=1)
	return inv_cdf(quantile_levels)

def calculate_confidence_limits(histograms,confidence_levels,bin_range):
	"""Calculates confidence limits from distributions represented as histograms.

	The band corresponding to each confidence level CL is given by a histogram's
	quantile levels (1-CL)/2 and 1-(1-CL)/2.

	Args:
		histograms(ndarray): Array of shape (nhist,nbins+2) containing nhist histograms with
			identical binning and including under- and overflow bins.
		confidence_levels(ndarray): Array of confidence levels to calculate.
		bin_range(ndarray): Array of length 2 containing the [min,max) range corresponding
			to the histogram contents in bins [1:-1].

	Returns:
		ndarray: Array of shape (2*ncl+1,nhist) where elements [i,j] and [-i,j] give the
			limits of the confidence band for confidence_levels[i] of histograms[j], and
			element [ncl,j] gives the median for histograms[j].
	"""
	nhist,nbins = histograms.shape
	lower_quantiles = np.sort(0.5*(1. - np.array(confidence_levels)))
	quantile_levels = np.concatenate((lower_quantiles,[0.5],1-lower_quantiles[::-1]))
	print quantile_levels
	limits = np.empty((len(quantile_levels),nhist))
	for ihist,hist in enumerate(histograms):
		limits[:,ihist] = quantiles(hist,quantile_levels,bin_range)
	return limits

def select_random_realizations(DH,DA,nll,n):
	"""Select random realizations of generated expansion histories.

	Args:
		DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
		DA(ndarray): Array of shape (nsamples,nz-1) of DA(z) values to use.
		nll(ndarray): Array of shape (npost,nsamples) containing the nll
			posterior weights to use.
		n(int): Number of random rows to return.

	Returns:
		ndarray: Array of shape (n,ncols) containing the selected random rows. Note that
			a row might be selected more than once.
	"""
	cdf = np.cumsum(np.exp(-nll))
	print cdf.shape
	pick = np.random.uniform(low=0.,high=cdf[-1],size=n)
	rows = cdf > pick
	print rows
