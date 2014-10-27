"""Gaussian random process generator.
"""

import numpy as np
import numpy.random

class SquaredExponentialGaussianProcess(object):
	"""Generates Gaussian process realizations using a squared-exponential kernel.

	The process is defined such that <s> = 0 and <s1*s2> = k(s1-s2) with the kernel
	k(ds) = h^2 exp(-ds^2/(2 sigma^2)).  The hyperparameters of this process are h
	and sigma, which establish the characteristic vertical and horizontal length
	scales, respectively.

	Args:
		hyper_h(float): vertical scaling hyperparameter.
		hyper_sigma(float): horizontal scaling hyperparameter.
	"""
	def __init__(self,hyper_h,hyper_sigma):
		self.hyper_h = hyper_h
		self.hyper_sigma = hyper_sigma

	def generate_samples(self,num_samples,svalues,seed=None):
		"""Generates random samples of our Gaussian process.

		Args:
			num_samples(int): Number of samples to generate.
			svalues(ndarray): Values of the evolution variable where the process
				will be sampled.
			seed(int): Random seed to use, or use default state if seed is None.

		Returns:
			ndarray: Array with shape (num_samples,len(svalues)) containing the
				generated samples.
		"""
		if seed is not None:
			numpy.random.seed(seed)
		# Evaluate the kernel for all pairs (s1,s2). This could be optimized to
		# evaluate only the s1 >= s2 pairs if necessary.
		s1,s2 = np.meshgrid(svalues,svalues,indexing='ij')
		ds = s1-s2
		covariance = self.hyper_h**2*np.exp(-ds**2/(2*self.hyper_sigma**2))
		# Sample this covariance with zero mean.
		mean = np.zeros_like(svalues)
		return np.random.multivariate_normal(mean,covariance,num_samples)
