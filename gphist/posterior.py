"""Expansion history posterior applied to distance functions.
"""

import math

import numpy as np
import numpy.linalg

class GaussianPdf(object):
	"""Represents a multi-dimensional Gaussian probability density function.

	Args:
		mean(ndarray): 1D array of length npar of mean values.
		covariance(ndarray): 2D symmetric positive definite covariance matrix
			of shape (npar,npar).

	Raises:
		ValueError: dimensions of mean and covariance are incompatible.
		LinAlgError: covariance matrix is not positive definite.
	"""
	def __init__(self,mean,covariance):
		# Check that the dimensions match or throw a ValueError.
		dimensions_check = mean.dot(covariance.dot(mean))
		# Check that the covariance is postive definite or throw a LinAlgError.
		posdef_check = numpy.linalg.cholesky(covariance)

		self.mean = mean
		self.covariance = covariance
		# Calculate the constant offset of -logL due to the normalization factors.
		self.norm = 0.5*len(mean)*np.log(2*math.pi) + 0.5*np.log(np.linalg.det(covariance))

	def get_nll(self,values):
		"""Calculates -logL for the PDF evaluated at specified values.

		The calculation is automatically broadcast over multiple value vectors.

		Args:
			values(ndarray): array values where the PDF should be evaluated, which can
				be a single vector of length npar or else a array of vectors.

		Returns:
			float: -log of the calculated likelihood.

		Raises:
			ValueError: values can not be broadcast together with our mean vector.
		"""
		# The next line will throw a ValueError if values cannot be broadcast.
		residuals = values - self.mean
		chisq = np.einsum('...ia,ab,...ib->...i',residuals,self.covariance,residuals)
		return self.norm + 0.5*chisq

class GaussianPdf1D(GaussianPdf):
	"""Represents a specialization of GaussianPdf to the 1D case.

	Args:
		central_value(float): central value of the 1D PDF.
		sigma(float): RMS spread of the 1D PDF.
	"""
	def __init__(self,central_value,sigma):
		mean = np.array([central_value])
		covariance = np.array([[sigma**2]])
		GaussianPdf.__init__(self,mean,covariance)

class LocalH0Posterior(GaussianPdf1D):
	"""Represents a posterior on the local value of H0 from Reiss 2011.
	"""
	def __init__(self):
		self.name = 'Local H0'
		GaussianPdf1D.__init__(self,74.8,3.1)

	def get_nll(self,DH,DC):
		"""Calculates -logL for the posterior applied to a set of expansion histories.

		The prior is applied to c/H(z=0).
		"""
		# Constant is speed of light in km/s. Indexing below is to get shape (n,1) for values.
		values = 299792.458/DH[:,0:1]
		return GaussianPdf1D.get_nll(self,values)
