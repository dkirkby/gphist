"""Cosmological evolution variables.

Variables are defined by a function s(z) that must be invertible and increase
monotonically from s(0) = 0 to s(zmax) = 1, where usually zmax ~ z*, the redshift
of last scattering. The function s(z) and its inverse z(s) are assumed to be
independent of the expansion history, for efficiency, but this could probably be
relaxed if necessary.
"""

import math

import numpy as np

def initialize(min_num_evol,num_evol_hist,num_samples,max_array_size):
    """Initialize evolution calculations.

    Args:
        min_num_evol(int): Minimum number of equally spaced evolution steps required.
        num_evol_hist(int): Number of downsampled equally spaced evolution steps required.
        num_samples(int): Number of prior samples to generate.
        max_array_size(float): Target size of arrays allocated for generated samples in gigabytes.

    Returns:
        tuple: Total number of equally spaced evolution steps to use, oversampling factor,
        and number of samples to generate per cycle.
    """
    # Calculate the oversampling factor required to reach or exceed min_num_evol steps.
    evol_oversampling = int(math.ceil((min_num_evol-1.)/(num_evol_hist-1.)))

    # Calculate the number of prior samples that fit within our memory budget
    # with this oversampling factor.
    num_evol = evol_oversampling*(num_evol_hist-1) + 1
    samples_per_cycle = int(math.floor(1e9*max_array_size/(16.*num_evol)))
    samples_per_cycle = min(num_samples,samples_per_cycle)

    return num_evol,evol_oversampling,samples_per_cycle

class LogScale(object):
    """Represents evolution using the logarithm of the scale factor a(t).

    LogScale evolution uses s(z) = log(1+z)/log(1+zmax) which is a scaled
    version of -log(a) where a = 1/(1+z) is the scale factor.

    Args:
        nsteps(int): Number of equally spaced steps to use between s=0
            and s=1. Actual number of steps can be larger to include values in zpost.
        oversampling(int): Oversampling factor relative to histogram sampling.
        zpost(ndarray): Array of redshifts where prior must be sampled in order to
            evaluate posterior constraints. The maximum value of zpost will be mapped
            to s=1.

    Raises:
        AssertionError: Invalid oversampling for nsteps. Used the initialize() method
            to ensure correct values, for which oversampling is a divisor of nsteps-1.
    """
    def __init__(self,nsteps,oversampling,zpost):
        assert (nsteps-1)%oversampling == 0,'invalid oversampling %d for nsteps %d' % (
            oversampling,nsteps)
        self.zmax = np.max(zpost)
        # Initialize equally spaced values of the evolution variable s.
        self.svalues = np.linspace(0.,1.,nsteps)
        # Calculate the corresponding zvalues.
        self.zvalues = self.z_of_s(self.svalues)
        # Downsample these zvalues.
        downsampled_zvalues = self.zvalues[::oversampling]
        # Add any values in zpost that are not already included.
        self.zvalues = np.unique(np.concatenate([self.zvalues,zpost]))
        self.svalues = self.s_of_z(self.zvalues)
        downsampled_zvalues = np.unique(np.concatenate([downsampled_zvalues,zpost]))
        # Find the indices of the downsampled values after adding adding zpost values.
        self.downsampled_indices = np.searchsorted(self.zvalues,downsampled_zvalues)
        assert np.array_equal(downsampled_zvalues,self.zvalues[self.downsampled_indices])
        # Initialize the quadrature coefficients needed by get_DC.
        delta = np.diff(self.zvalues)/np.diff(self.svalues)/np.log(1+self.zmax)
        self.quad_coef1 = 1 + self.zvalues[:-1] - delta
        self.quad_coef2 = 1 + self.zvalues[1:] - delta

    def s_of_z(self,z):
        """Evaluate the function s(z).

        Automatically broadcasts over an input array.

        Args:
            z(ndarray): Array of redshifts for calculating s(z).

        Returns:
            ndarray: Array of evolution variable values s(z).
        """
        return np.log(1+z)/np.log(1+self.zmax)

    def z_of_s(self,s):
        """Evaluate the inverse function z(s).

        Automatically broadcasts over an input array.

        Args:
            s(ndarray): Array of evolution variable values for calculating z(s).

        Returns:
            ndarray: Array of redshifts z(s).
        """
        return np.power(1+self.zmax,s) - 1

    def get_DC(self,DH):
        """Converts Hubble distances DH(z) to comoving distances DC(z).

        Performs the integral DC(z) = Integrate[DH(zz),{zz,0,z}] using linear
        interpolation of DH in s.

        Args:
            DH(ndarray): 2D array of tabulated Hubble distances DH with shape
                (nsamples,nsteps). DH[i,j] is interpreted as the distance to
                zvalues[j] for sample i.

        Returns:
            ndarray: 2D array of tabulated comoving distances DC with the same
                shape as the input DH array. The [i,j] value gives DC at
                zvalues[j].
        """
        # Initialize result array.
        DC = np.empty_like(DH)
        # Set DC(z=0) = 0.
        DC[:,0] = 0.
        # Tabulate values of DH(z[i+1]) - DH(z[i]) in DC
        DC[:,1:] = self.quad_coef2*DH[:,1:] - self.quad_coef1*DH[:,:-1]
        # Reconstruct DC for z > 0.
        np.cumsum(DC[:,1:],axis=1,out=DC[:,1:])
        return DC
