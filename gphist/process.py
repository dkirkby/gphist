"""Gaussian random process generator.
"""

import numpy as np

class HyperParameterLogGrid(object):
	"""Defines a log-spaced grid of hyperparameter values.

	Args:
		n_h(int): Number of grid points covering hyperparameter h.
		h_min(float): Minimum grid value of hyperparameter h.
		h_max(float): Maximum grid value of hyperparameter h.
		n_sigma(int): Number of grid points covering hyperparameter sigma.
		sigma_min(float): Minimum grid value of hyperparameter sigma.
		sigma_max(float): Maximum grid value of hyperparameter sigma.
	"""
	def __init__(self,n_h,h_min,h_max,n_sigma,sigma_min,sigma_max):
		h_ratio = np.power(h_max/h_min,1./(n_h-1))
		sigma_ratio = np.power(sigma_max/sigma_min,1./(n_sigma-1))
		self.h = h_min*np.power(h_ratio,np.arange(n_h))
		self.sigma = sigma_min*np.power(sigma_ratio,np.arange(n_sigma))
		self.n_h = n_h
		# Initialize log-spaced bin edges to support matplotlib pcolormesh.
		self.h_edges = h_min*np.power(h_ratio,np.arange(n_h+1)-0.5)
		self.sigma_edges = sigma_min*np.power(sigma_ratio,np.arange(n_sigma)-0.5)
		self.h_edges[0],self.h_edges[-1] = self.h[0],self.h[-1]
		self.sigma_edges[0],self.sigma_edges[-1] = self.sigma[0],self.sigma[-1]

	def decode_index(self,index):
		"""Decode a flattened grid index.

		Args:
			index(int): Flattened index in the range [0:n_h*n_sigma].

		Returns:
			tuple: h,sigma index values.
		"""
		return index%self.n_h,index//self.n_h

	def get_values(self,index):
		"""Lookup hyperparameter values on the grid.

		Args:
			index(int): Flattened index in the range [0:n_h*n_sigma].

		Returns:
			tuple: Values of h,sigma at the specified grid point.
		"""
		i_h,i_sigma = self.decode_index(index)
		return self.h[i_h],self.sigma[i_sigma]

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

	def generate_samples(self,num_samples,svalues,random_state=None):
		"""Generates random samples of our Gaussian process.

		Args:
			num_samples(int): Number of samples to generate.
			svalues(ndarray): Values of the evolution variable where the process
				will be sampled.
			random_state(numpy.RandomState): Random state to use, or use default
				state if None.

		Returns:
			ndarray: Array with shape (num_samples,len(svalues)) containing the
				generated samples.
		"""
		# Evaluate the kernel for all pairs (s1,s2). This could be optimized to
		# evaluate only the s1 >= s2 pairs if necessary.
		s1,s2 = np.meshgrid(svalues,svalues,indexing='ij')
		ds = s1-s2
		covariance = self.hyper_h**2*np.exp(-ds**2/(2*self.hyper_sigma**2))
		# Sample this covariance with zero mean.
		mean = np.zeros_like(svalues)
		# Fall back to the default random generator if necessary.
		generator = random_state if random_state else np.random
		return generator.multivariate_normal(mean,covariance,num_samples)
