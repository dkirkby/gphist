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
	"""

	def __init__(self,nsteps,zstar):
		"""Initializes a LogScale evolution variable.

		Args:
			nsteps(int): number of equally spaced steps to use between s=0
				and s=1.
			zstar(float): last scattering redshift corresponding to s=1.
		"""
		self.zstar = zstar
		# Initialize equally spaced values of the evolution variable s.
		self.svalues = np.linspace(0.,1.,nsteps)
		# Calculate the corresponding zvalues for the specified z*.
		self.zvalues = np.power(1+zstar,self.svalues) - 1
		# Initialize the quadrature coefficients needed by get_DC.
		delta = np.diff(self.zvalues)/np.diff(self.svalues)/np.log(1+zstar)
		self.quad_coef1 = 1 + self.zvalues[:-1] - delta
		self.quad_coef2 = 1 + self.zvalues[1:] - delta

	def get_DC(self,DH):
		"""Converts Hubble distances DH(z) to comoving distances DC(z).

		Performs the integral DC(z) = Integrate[DH(zz),{zz,0,z}] using linear
		interpolation of DH in s.

		Args:
			DH(ndarray): 2D array of tabulated Hubble distances DH with shape
				(nsamples,nsteps).

		Returns:
			ndarray: 2D array of tabulated comoving distances DC with shape
				(nsamples,len(s)).
		"""
		# Tabulate values of DC(z[i+1]) - DC(z[i]).
		deltaDC = self.quad_coef2*DH[:,1:] - self.quad_coef1*DH[:,:-1]
		# Reconstruct DC.
		DC = np.empty_like(DH)
		DC[:,0] = 0.
		np.cumsum(deltaDC,axis=1,out=DC[:,1:])
		return DC
