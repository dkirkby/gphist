"""Cosmological evolution variables.

Variables are defined by a function s(z) that must be invertible and increase
monotonically from s(0) = 0 to s(z*) = 1, where z* is the redshift of last scattering.
The function s(z) and its inverse can, in general, depend on the expansion history
and geometry, as encapsulated in the distance functions DH(z) and DA(z).
"""

import numpy as np

class LogScale(object):
	"""Represents evolution using the logarithm of the scale factor a(t).

	LogScale evolution uses s(z) = log(1+z)/log(1+z*) which is a scaled
	version of -log(a) where a = 1/(1+z) is the scale factor.

	Args:
		nsteps(int): number of equally spaced steps to use between s=0
			and s=1.
		zstar(float): last scattering redshift corresponding to s=1.
	"""

	def __init__(self,nsteps,zstar):
		self.zstar = zstar
		# Initialize equally spaced values of the evolution variable s.
		self.svalues = np.linspace(0.,1.,nsteps)
		# Calculate the corresponding zvalues for the specified z*.
		self.zvalues = np.power(1+zstar,self.svalues) - 1
		# Initialize the quadrature coefficients needed by get_DC.
		delta = np.diff(self.zvalues)/np.diff(self.svalues)/np.log(1+zstar)
		self.quad_coef1 = 1 + self.zvalues[:-1] - delta
		self.quad_coef2 = 1 + self.zvalues[1:] - delta

	def s_of_z(self,z):
		"""Evaluates the function s(z).

		Automatically broadcasts over a redshift array.

		Args:
			z(ndarray): redshift where evolution variable s should be evaluated.

		Returns:
			ndarray: value of evolution variable s(z).
		"""
		return np.log(1+z)/np.log(1+self.zstar)

	def get_DC(self,DH):
		"""Converts Hubble distances DH(z) to comoving distances DC(z).

		Performs the integral DC(z) = Integrate[DH(zz),{zz,0,z}] using linear
		interpolation of DH in s. Note that the returned array has a different
		shape from the input array (one less column) since DC(z=0) = 0 is
		not included.

		Args:
			DH(ndarray): 2D array of tabulated Hubble distances DH with shape
				(nsamples,nsteps). DH[i,j] is interpreted as the distance to
				zvalues[j] for sample i.

		Returns:
			ndarray: 2D array of tabulated comoving distances DC with shape
				(nsamples,nsteps-1). The [i,j] value gives DC at zvalues[j+1].
				The value DC(z=0) = 0 is not included.
		"""
		# Tabulate values of DC(z[i+1]) - DC(z[i]).
		deltaDC = self.quad_coef2*DH[:,1:] - self.quad_coef1*DH[:,:-1]
		# Reconstruct DC.
		return np.cumsum(deltaDC,axis=1)
