"""Cosmological distance model.
"""

import numpy as np

import gphist # for clight_km_per_sec

def convert_DC_to_DA(DH,DC,omega_k):
    """Applies curvature to convert DC(z) into DA(z).

    Args:
        DH(ndarray): input array with shape (nh,nz) of nh comoving distance
            functions DC(z) each tabulated at nz redshifts. Only the slice
            DH[:,0] corresponding to z=0 is actually used.
        DC(ndarray): input array with shape (nh,nz) of nh comoving distance
            functions DC(z) each tabulated at nz redshifts starting with z=0.
            The input values are overwritten with DA values.
        omega_k(float): curvature parameter Omega_k.

    Returns:
        ndarray: a reference to the input array DC which now contains
            values of DA.
    """
    if omega_k < 0:
        w = DH[:,0]/np.sqrt(-omega_k)
        DC[:] = w*np.sin(DC/w)
    elif omega_k > 0:
        w = DH[:,0]/np.sqrt(+omega_k)
        DC[:] = w*np.sinh(DC*w)
    return DC

def get_acceleration(z,DH):
    """Calculates the cosmic acceleration H(z)/(1+z).

    Args:
        z(ndarray): Array of redshifts where DH is tabulated.
        DH(ndarray): Array of Hubble distances DH(z) = c/H(z). Must have
            the same last dimension as z.

    Returns:
        ndarray: Array of calculated H(z)/(1+z) values with the same
            shape as the input DH array.

    Raises:
        ValueError: Arrays z and DH do not have the same last dimension.
    """
    return gphist.clight_km_per_sec/DH/(1+z)

def fiducial_DH(z):
    """Evaluates the fiducial cosmology distance function DH(z).

    The fiducial model is derived from the Planck+WP best fit to LCDM in Ade 2013.
    The form below was obtained in Mathematica using::

        Needs["DeepZot`CosmoTools`"]
        createCosmology[cosmo]
        CForm[hubbleDistance[cosmo]/Hratio[cosmo][z]]

    The DeepZot mathematica packages are at https://github.com/deepzot/mathpkg

    Args:
        z(ndarray): Array of redshifts where DH(z) = c/H(z) should be evaluated.

    Returns:
        ndarray: Array of values of DH(z) = c/H(z) in Mpc.
    """
    zp1 = 1+z
    denom_squared = 0.681619598847103 + zp1**3*(0.3168651267559463 +
        zp1*(0.000055024838575896826 + 0.00001268815716056235*(
            2 + np.power(1 + 6648.9902742842605*zp1**(-1.8614683763259083),
                0.5372103081190989))))
    return 4471.8445405727925/np.sqrt(denom_squared)

class HubbleDistanceModel(object):
    """Models expansion history as multiplicative correction to the fiducial DH(z).

    Args:
        evol: evolution parameter to use, which must have a zvalues attribute
            and implement a get_DC method.
    """
    def __init__(self,evol):
        # Tabulate values of DH(z) for the fiducial model.
        self.DH0 = fiducial_DH(evol.zvalues)
        # Calculate the corresponding DC values.
        self.DC0 = evol.get_DC(self.DH0[np.newaxis,:])[0]

    def get_DH(self,samples):
        """Build expansion histories from Gaussian process samples.

        Each sample gamma(z) generates a Hubble distance function
        DH(z) = DH0(z)*exp(gamma(z)).

        Args:
            samples(ndarray): 2D array with shape (num_samples,num_steps)
                of num_samples samples gamma(z) tabulated at the num_steps
                redshifts z given in zvalues.

        Returns:
            ndarray: 1D array with shape (num_steps,) of tabulated DH(z) values.
        """
        return self.DH0*np.exp(samples)
