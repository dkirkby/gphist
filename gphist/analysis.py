"""Analysis and plotting functions.
"""

import numpy as np
import scipy.interpolate
import scipy.stats

def get_delta_chisq(confidence_levels=(0.6827,0.9543,0.9973),num_dof=2):
    """
    Calculate delta(chisq) values.

    The default confidence_level values correspond to the enclosed probabilities
    of 1,2,3-sigmas for a 1D Gaussian pdf.

    Args:
    	confidence_levels: Iterable of confidence levels that should be converted to
    		delta(chisq) values, which do not need to be sorted. In case this argument
    		is a numpy array, the calculation will be broadcast over it.
    	num_dof(int): Number of degrees of freedom to assume for the calculation.

    Returns:
    	ndarray: Array of delta(chisq) values with the same shape as the input
    		confidence_levels.
    """
    return scipy.stats.chi2.isf(1-np.array(confidence_levels),df=num_dof)

def calculate_posteriors_nlp(zprior,DH,DA,posteriors):
	"""Calculate -log(prob) for each combination of posterior and prior sample.

	Args:
		zprior(ndarray): Redshifts where prior is sampled, in increasing order.
		DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
		DA(ndarray): Array of shape (nsamples,nz) of DA(z) values to use.
		posteriors(list): List of posteriors to use. Each posterior should
			inherit from the posterior.Posterior base class.

	Returns:
		ndarray: An array of shape (npost,nsamples) containing the nlp values
			calculated for each posterior independently.
	"""
	nsamples = len(DH)
	npost = len(posteriors)
	nlp = np.empty((npost,nsamples))
	for ipost,post in enumerate(posteriors):
		nlp[ipost] = post.get_nlp(zprior,DH,DA)
	return nlp

def get_bin_indices(data,num_bins,min_value,max_value):
	"""Create array of integer bin indices in preparation for histogramming.

	Results are in a format suitable for using with numpy.bincount. This function
	only handles uniform binning, but is generally faster than numpy.histogram in
	this case, especially when this method is broadcast over data that is destined
	for multiple histograms.

	Args:
		data(ndarray): Array of data to histogram.
		num_bins(int): Number of equally spaced bins to use for each histogram.
		min_value(float): Values less than this are accumulated in an underflow bin.
		max_value(float): Values greater than equal to this are accumulated in an
			overflow bin.

	Returns:
		ndarray: Array of integer bin indices with the same shape as data. Each
			input data value is replaced with its corresponding bin index in the
			range [1:num_bins+1] or 0 if value < min_value or num_bins+1 if
			value >= max_value.

	Raises:
		ValueError: num_bins <= 0 or min_value >= max_value.
	"""
	if num_bins <= 0:
		raise ValueError('num_bins <= 0')
	if min_value >= max_value:
		raise ValueError('min_value >= max_value')
	bin_indices = np.floor((data-min_value)/(max_value-min_value)*num_bins).astype(int)+1
	# Combine index < 1 into the underflow bin [0] and index > nbins+1 into the
	# overflow [nbins+1].
	return np.maximum(np.minimum(bin_indices,num_bins+1),0)

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

def calculate_histograms(DH,DH0,DA,DA0,de_evol,de0_evol,nlp,num_bins,min_value,max_value):
	"""Build histograms for all permutations of posterior weightings.

	The undefined ratio DA(z=0)/DA0(z=0) is never evaluated.

	Args:
		DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
		DH0(ndarray): Array of shape (nz,) used to normalize each DH(z).
		DA(ndarray): Array of shape (nsamples,nz) of DA(z) values to use.
		DA0(ndarray): Array of shape (nz,) used to normalize each DA(z).
		de_evol(ndarray): Array of shape (nde,nsamples,nz) of nde dark-energy
			evolution histories, or None if they have not been calculated.
		de0_evol(ndarray): Array of shape (nde,nz) of nde fiducial dark-energy
			evolution histories, used for normalization.
		nlp(ndarray): Array of shape (npost,nsamples) containing the nlp
			posterior weights to use.
		num_bins(int): Number of equally spaced bins to use for each histogram.
		min_value(float): Values less than this are accumulated in an underflow bin.
		max_value(float): Values greater than equal to this are accumulated in an
			overflow bin.

	Returns:
		tuple: Arrays of histograms for DH/DH0, DA/DA0, de_evol/de0_evol with shapes
			(nperm,nz,num_bins+2), (nperm,nz-1,num_bins+2), (nde,nperm,nz,num_bins+2),
			respectively, where nperm = 2**npost. There are no DA/DA0 histograms for z=0,
			hence the nz vs nz-1 dimensions. The mapping between permutations and
			the permutation index is given by the binary representation of the index.
			For example, iperm = 5 = 2^0 + 2^2 combines posteriors 0 and 2.

	Raises:
		AssertionError: Unexpected sizes of DH0,DA0,DA,de_evol,de0_evol,nlp.
	"""
	npost = len(nlp)
	nsamples,nz = DH.shape
	nde = len(de0_evol)
	# Check sizes.
	assert DH0.shape == (nz,),'Unexpected DH0.shape'
	assert DA0.shape == (nz,),'Unexpected DA0.shape'
	assert DA.shape == (nsamples,nz),'Unexpected DA.shape'
	assert de0_evol.shape == (nde,nz), 'Unexpected de0_evol shape'
	assert de_evol is None or de_evol.shape == (nde,nsamples,nz), 'Unexpected de_evol shape'
	assert nlp.shape == (npost,nsamples),'Unexpected nlp.shape'
	# Rescale DH,DA by DH0,DA0. We drop the z=0 bin for the DA ratio to avoid 0/0.
	DH_ratio = DH/DH0
	DA_ratio = DA[:,1:]/DA0[1:]
	# Initialize posterior permutations.
	nperm = 2**npost
	perms = get_permutations(npost)
	# Allocate output arrays for the histogram bin values.
	DH_hist = np.empty((nperm,nz,num_bins+2))
	DA_hist = np.empty((nperm,nz-1,num_bins+2))
	de_hist = np.empty((nde,nperm,nz-1,num_bins+2)) if de_evol is not None else None
	# Calculate bin indices (after swapping nsamples,nz axes).
	DH_bin_indices = get_bin_indices(DH_ratio.T,num_bins,min_value,max_value)
	DA_bin_indices = get_bin_indices(DA_ratio.T,num_bins,min_value,max_value)
	if de_evol is not None:
		de_ratio = de_evol[:,:,:-1]/de0_evol[:,np.newaxis,:-1]
		# Calculate bin indices (after swapping nsamples,nz axes).
		de_bin_indices = get_bin_indices(np.swapaxes(de_ratio,1,2),
			num_bins,min_value,max_value)
	# Loop over permutations.
	for iperm,perm in enumerate(perms):
		# Calculate weights for this permutation.
		perm_nlp = np.sum(nlp[perm],axis=0)  # Returns zero when perm entries all False.
		perm_weights = np.exp(-perm_nlp)
		for ihist in range(nz):
			# Build histograms of DH/DH0 for each redshift slice.
			DH_hist[iperm,ihist] = np.bincount(
				DH_bin_indices[ihist],weights=perm_weights,minlength=num_bins+2)
			# Build histograms of DA/DA0 for each redshift slice (skipping z=0).
			if ihist > 0:
				DA_hist[iperm,ihist-1] = np.bincount(
					DA_bin_indices[ihist-1],weights=perm_weights,minlength=num_bins+2)
			# Build histograms of each dark-energy evolution variable (skipping zmax).
			if de_evol is not None and ihist < nz-1:
				for ide in range(nde):
					de_hist[ide,iperm,ihist] = np.bincount(de_bin_indices[ide,ihist],
						weights=perm_weights,minlength=num_bins+2)
	return DH_hist,DA_hist,de_hist

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

	Returns:
		ndarray: Array of estimated abscissa values for each input level. Each value will
			be in the range specified by bin_range. Any values that should be outside of
			this range will be clipped.

	Raises:
		ValueError: bin_range does not have 2 elements.
	"""
	# Reconstruct the histogram binning.
	num_bins = len(histogram)-2
	min_value,max_value = bin_range # Raises ValueError if there are not exactly 2 values to unpack.
	bin_edges = np.linspace(min_value,max_value,num_bins+1,endpoint=True)
	# Build the cummulative distribution function, including the under/overflow bins.
	cdf = np.cumsum(histogram)
	cdf /= cdf[-1]
	# Skip almost empty bins so that CDF values are increasing for inverse interpolation.
	use = np.diff(cdf) > threshold
	use[0] = True
	# Linearly interpolate CDF levels to estimate the corresponding bin values.
	inv_cdf = scipy.interpolate.InterpolatedUnivariateSpline(
		cdf[use[:-1]],bin_edges[use[:-1]],k=1)
	levels = inv_cdf(quantile_levels)
	# Perform clipping for levels outside our interpolation range.
	levels[quantile_levels < cdf[0]] = min_value
	levels[quantile_levels >= cdf[-2]] = max_value
	return levels

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
			element [ncl,j] gives the median for histograms[j]. Values that lie outside of
			bin_range will be clipped.
	"""
	nhist,nbins = histograms.shape
	lower_quantiles = np.sort(0.5*(1. - np.array(confidence_levels)))
	quantile_levels = np.concatenate((lower_quantiles,[0.5],1-lower_quantiles[::-1]))
	limits = np.empty((len(quantile_levels),nhist))
	for ihist,hist in enumerate(histograms):
		limits[:,ihist] = quantiles(hist,quantile_levels,bin_range)
	return limits

def select_random_realizations(DH,DA,nlp,num_realizations,
	random_state=None,print_warnings=True):
	"""Select random realizations of generated expansion histories.

	Args:
		DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
		DA(ndarray): Array of shape (nsamples,nz) of DA(z) values to use.
		nlp(ndarray): Array of shape (npost,nsamples) containing the nlp
			posterior weights to use.
		num_realizations(int): Number of random rows to return.
		random_state(numpy.RandomState): Random state to use, or use default
			state if None.
		print_warnings(bool): Print a warning for any posterior permutation
			whose selected realizations include repeats.

	Returns:
		tuple: Arrays of random realizations of DH and DA, with shape
			(nperm,num_realizations,nz) where nperm = 2**npost is the total
			number of posterior permutations. Note that a realization might
			be selected more than once. Use the print_warnings argument to
			flag this.

	Raises:
		AssertionError: Unexpected sizes of DH,DA, or nlp.
	"""
	nsamples,nz = DH.shape
	npost = len(nlp)
	# Check sizes.
	assert DA.shape == (nsamples,nz)
	assert nlp.shape == (npost,nsamples)
	# Allocate result arrays.
	nperm = 2**npost
	DH_realizations = np.empty((nperm,num_realizations,nz))
	DA_realizations = np.empty((nperm,num_realizations,nz))
	# Fall back to the default random generator if necessary.
	generator = random_state if random_state else np.random
	# Generate a random CDF value for each realization.
	random_levels = generator.uniform(low=0.,high=1.,size=num_realizations)
	# Loop over posterior permutations.
	perms = get_permutations(npost)
	for iperm,perm in enumerate(perms):
		# Calculate weights for this permutation.
		perm_nlp = np.sum(nlp[perm],axis=0)  # Returns zero when perm entries are all False.
		perm_weights = np.exp(-perm_nlp)
		perm_cdf = np.cumsum(perm_weights)
		perm_cdf /= perm_cdf[-1]
		perm_rows = np.argmax(perm_cdf > random_levels[:,np.newaxis],axis=1)
		if print_warnings:
			num_unique = np.unique(perm_rows).size
			if num_unique < num_realizations:
				print 'WARNING: only %d of %d realizations are unique for permutation %d' % (
					num_unique,num_realizations,iperm)
		DH_realizations[iperm] = DH[perm_rows]
		DA_realizations[iperm] = DA[perm_rows]
	return DH_realizations,DA_realizations
