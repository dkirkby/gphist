"""Expansion history posterior applied to distance functions.
"""

import math
from abc import ABCMeta,abstractmethod

import numpy as np
import numpy.linalg
import scipy.interpolate
import astropy.constants

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
		#print 'the shape of the mean is'+str(mean.shape)
		#print 'the shape of the covariance is'+str(covariance.shape)
		#dimensions_check = mean.dot(covariance.dot(mean))
		# Check that the covariance is postive definite or throw a LinAlgError.
        #posdef_check = numpy.linalg.cholesky(covariance[:,:,0])

		self.mean = mean
		if len(covariance.shape) > 2: #this assumes there are exactly 3 covariance matrices for 3 redshifts
			#print 'this should only happen once for the boss data'			
			temp_icov = np.linalg.inv(covariance[:,:,0])
			for i in range(1,covariance.shape[2]):
			    stack_temp = np.linalg.inv(covariance[:,:,i])
			    temp_icov = np.dstack([temp_icov,stack_temp])
			self.icov = temp_icov
			temp_norm = 0.5*mean.size*np.log(2*math.pi) + 0.5*(np.log(np.linalg.det(covariance[:,:,0])))
			for i in range(1,covariance.shape[2]):
			    temp_norm+=0.5*(np.log(np.linalg.det(covariance[:,:,i])))
			self.norm = temp_norm
		else:
			#print 'this should happen 5 times not 6'
			#print np.linalg.det(covariance)
			self.icov = np.linalg.inv(covariance)
			self.norm = 0.5*mean.size*np.log(2*math.pi) + 0.5*np.log(np.linalg.det(covariance))
		#print self.mean.shape
		#print self.icov.shape
		# Calculate the constant offset of -log(prob) due to the normalization factors.

	def get_nlp(self,values):
		"""Calculates -log(prob) for the PDF evaluated at specified values.

		The calculation is automatically broadcast over multiple value vectors.

		Args:
			values(ndarray): Array of values where the PDF should be evaluated with
				shape (neval,ndim) where ndim is the dimensionality of the PDF and
				neval is the number of points where the PDF should be evaluated.
			more precisely, it has shape (nsamples,ntype,nzposterior)
			nsamples is the number of samples requested
			ntype is the number of types of posteriors, types being DH, DA or mu
			nzposterior is the number of redshifts in a given posterior

		Returns:
			float: Array of length neval -log(prob) values calculated at each input point.

		Raises:
			ValueError: Values can not be broadcast together with our mean vector.
		"""
		# The next line will throw a ValueError if values cannot be broadcast.
		#print self.name
		#print 'in order: values, mean, residual, icov'
		#print 'the values are'
		#print values
		#print 'the mean is'
		#print self.mean
		#print self.mean
		residuals = values - self.mean
		#print residuals
		a = residuals.shape
		#print self.icov
		#print 'the shape of the covariance matrix is'+str(self.icov.shape)
		#print a
		#print a[2]
		#chisq = np.tensordot()
        #a[2] is ntype; the difference between these cases which dimension the covariance matrix is for
        # ALL OF THESE RESIDUALS SHOULD BE OF THE FORM (NSAMPLE,NZ,NTYPE)
		if a[1]>1 and a[2]==1:   # should correspond to just SN data
			#print 'Type 1'
			#print 'the values with shape '+str(values.shape) +' are'
			#print values
			#print 'the mean with shape '+str(self.mean.shape) +'is'
			#print self.mean
			#print 'the residuals with shape '+str(residuals.shape) +' are'
			#print residuals
			#print 'the icov with shape '+str(self.icov.shape)+ ' is'
			#print self.icov
			chisq = np.einsum('...ijk,jl,...ilk->...i',residuals,self.icov,residuals)
			#print 'the normalization is'
			#print self.norm
			#print 'the chisq is '
			#print 0.5*chisq
			#print 'the total log prob is'
			#print self.norm + 0.5*chisq
		elif a[2]>1 and a[1]>1:  #should correspond to just BOSS 2016
			#print 'Type 2'
			#print 'the values with shape '+str(values.shape) +' are'
			#print values
			#print 'the mean with shape '+str(self.mean.shape) +'is'
			#print self.mean
			#print 'the residuals with shape '+str(residuals.shape) +' are'
			#print residuals
			#print 'the icov with shape '+str(self.icov.shape)+ ' is'
			#print self.icov
			chisq = np.einsum('...ijk,klj,...ijl->...i',residuals,self.icov,residuals)
		else:
			#print 'Type 3'
			#print 'the values with shape '+str(values.shape) +' are'
			#print values
			#print 'the mean with shape '+str(self.mean.shape) +'is'
			#print self.mean
			#print 'the residuals with shape '+str(residuals.shape) +' are'
			#print residuals
			#print 'the icov with shape '+str(self.icov.shape)+ ' is'
			#print self.icov
			chisq = np.einsum('...ijk,kl,...ijl->...i',residuals,self.icov,residuals)
		#print chisq
		#print 'the normalization is'
		#print self.norm
		#print 'the chisq is '
		#print 0.5*chisq
		return self.norm + 0.5*chisq

class GaussianPdf1D(GaussianPdf):
	"""Represents a specialization of GaussianPdf to the 1D case.

	Args:
		central_value(float): Central value of the 1D PDF.
		sigma(float): RMS spread of the 1D PDF.
	"""
	def __init__(self,central_value,sigma):
		mean = np.array([central_value])
		mean = mean[:,np.newaxis]
		covariance = np.array([[sigma**2]])
		GaussianPdf.__init__(self,mean,covariance)

	def get_nlp(self,values):
		"""Calculates -log(prob) for the PDF evaluated at specified values.

		Args:
			values(ndarray): Array of values where the PDF should be evaluated with
				length neval.

		Returns:
			float: Array of length neval -log(prob) values calculated at each input point.
		"""
		return GaussianPdf.get_nlp(self,values[:,np.newaxis])

class GaussianPdf2D(GaussianPdf):
	"""Represents a specialization of GaussianPdf to the 2D case.

	Args:
		x1(float): Central value of the first parameter.
		x2(float): Central value of the second parameter.
		sigma1(float): RMS spread of the first parameter.
		sigma2(float): RMS spread of the second parameter.
		rho12(float): Correlation coefficient between the two parameters. Must be
			between -1 and +1.
	"""
	def __init__(self,x1,sigma1,x2,sigma2,rho12):
		mean = np.array([x1,x2])
		cov12 = sigma1*sigma2*rho12
		covariance = np.array([[sigma1**2,cov12],[cov12,sigma2**2]])
		mean = mean[np.newaxis,:]#see the CMB posterior class
		GaussianPdf.__init__(self,mean,covariance)

class Posterior(object):
	"""Posterior constraint on DH,DA at a fixed redshift.

	This is an abstract base class and subclasses must implement the constraint method.

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
	"""
	__metaclass__ = ABCMeta
	def __init__(self,name,zpost):
		self.name = name
		self.zpost = zpost

	@abstractmethod
	def constraint(self,DHz,DAz,muz):
		"""Evaluate the posterior constraint given values of DH(zpost) and DA(zpost).

		Args:
			DHz(ndarray): Array of DH(zpost) values.
			DAz(ndarray): Array of DA(zpost) values with the same shape as DHz.

		Returns:
			nlp(ndarray): Array of -log(prob) values with the same shape as DHz and DAz.
		"""
		pass

	def get_nlp(self,zprior,DH,DA,mu):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		The posterior is applied to c/H(z=0).

			zprior(ndarray): Redshifts where prior is sampled, in increasing order.
			DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
			DA(ndarray): Array of shape (nsamples,nz) of DA(z) values to use.

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.

		Raises:
			AssertionError: zpost is not in zprior.
		"""
		#iprior = np.argmax(zprior==self.zpost)
		iprior = np.where(np.in1d(zprior,self.zpost))[0] #for whatever reason np.where returns a tuple of an array so thats why there is the [0] after
		#iprior_test = np.append(iprior)
		#print self.name
		#print iprior[0]
		#print zprior[iprior[0]]
		#print iprior_test
		#assert zprior[iprior[0]] == self.zpost,'zpost is not in zprior'
		#DHz = DH[:,iprior[0]]
		#DAz = DA[:,iprior[0]]
		#muz = mu[:,iprior[0]]
		#DHz_test = DH[:,iprior][:,0]
		#DAz_test = DA[:,iprior][:,0]
		#muz_test = mu[:,iprior][:,0]
		#DHz_test2 = DH[:,iprior]
		#DAz_test2 = DA[:,iprior]
		#muz_test2 = mu[:,iprior]
		#DHz_test2 = DH[:,iprior_test]
		#DAz_test2 = DA[:,iprior_test]
		#muz_test2 = mu[:,iprior_test]
		#DHz = DH[:,iprior][:,0] #curious, this implementation raises the assert error
		#DAz = DA[:,iprior][:,0]
		#muz = mu[:,iprior][:,0]
		DHz = DH[:,iprior]
		DAz = DA[:,iprior]
		muz = mu[:,iprior]# these should be of the form (nsample,nz)
		#print DHz.shape		
		#print DHz_test.shape
		#print DHz_test2.shape
		#assert id(DH)==id(DHz.base)
		#assert id(DH)==id(DHz_test.base)
		#assert id(DA)==id(DAz.base)
		return self.constraint(DHz,DAz,muz)

class LocalH0Posterior(Posterior):
	"""Posterior constraint on the value of H0 determined from local measurements.

	Args:
		name(str): Name to associate with this posterior.
		H0(float): Central value of H(z=0).
		H0_error(float): RMS error on H(z=0).
	"""
	def __init__(self,name,H0,H0_error):
		self.pdf = GaussianPdf1D(H0,H0_error)
		Posterior.__init__(self,name,0.)

	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		The posterior is applied to c/H(z=0).

			DHz(ndarray): Array of DH(z=0) values to use.
			DAz(ndarray): Array of DA(z=0) values to use (will be ignored).

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		clight_km_per_sec = astropy.constants.c.to('km/s').value
		#print 'the shape of DHz is'+str(DHz.shape)
		#DHz = DHz[:,:,np.newaxis]
		#print DHz.shape
		#a= clight_km_per_sec/DHz
		#print a.shape
		return self.pdf.get_nlp(clight_km_per_sec/DHz)

class DHPosterior(Posterior):
	"""Posterior constraint on DH(z).

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
		DH(float): Central value of DH(z).
		DH_error(float): RMS error on DH(z).
	"""
	def __init__(self,name,zpost,DH,DH_error):
		self.pdf = GaussianPdf1D(DH,DH_error)
		Posterior.__init__(self,name,zpost)

	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		Args:
			DHz(ndarray): Array of DH(zpost) values to use.
			DAz(ndarray): Array of DA(zpost) values to use (will be ignored).

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		#DHz = DHz[:,:,np.newaxis]
		return self.pdf.get_nlp(DHz)

class DAPosterior(Posterior):
	"""Posterior constraint on DA(z).

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
		DA(float): Central value of DA(z).
		DA_error(float): RMS error on DA(z).
	"""
	def __init__(self,name,zpost,DA,DA_error):
		self.pdf = GaussianPdf1D(DA,DA_error)
		Posterior.__init__(self,name,zpost)

	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		Args:
			DHz(ndarray): Array of DH(zpost) values to use (will be ignored).
			DAz(ndarray): Array of DA(zpost) values to use.

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		#DAz = DAz[:,:,np.newaxis]
		print DAz.shape
		return self.pdf.get_nlp(DAz)

class CMBPosterior(Posterior):
	"""Posterior constraint on DH(zref) and DA(zref) from CMB with zpost ~ z*.

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift where posterior should be evaluated.
		DH(float): Value of DH(zref) at zref=evol.zvalues[-1].
		DA1pz(float): Value of DA(zref)/(1+zref) at zref=evol.zvalues[-1].
		cov11(float): Variance of DH(zref).
		cov12(float): Covariance of DH(zref) and DA(zref)/(1+zref).
		cov22(float): Variance of DA(zref)/(1+zref).
	"""
	def __init__(self,name,zpost,DH,DA1pz,cov11,cov12,cov22):
		mean = np.array([DH,DA1pz*(1+zpost)])
		cov12 *= (1+zpost)
		cov22 *= (1+zpost)**2
		covariance = np.array([[cov11,cov12],[cov12,cov22]])
		#print 'the covariance matrix for the CMB is'
		#print covariance
		mean = mean[np.newaxis,:]#trying to make all the means of the form (nz,ntype) since the values will be of the form (nsample,NZ,NTYPE)
		#print 'the shape of the CMB mean is'
		#print mean.shape
		self.pdf = GaussianPdf(mean,covariance)
		Posterior.__init__(self,name,zpost)

	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		Args:
			DHz(ndarray): Array of DH(zpost) values to use.
			DAz(ndarray): Array of DA(zpost) values to use.

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		values = np.dstack([DHz,DAz]) #dstack since DH(A)z are of the form (nsamples,nz) and we want (nsamples,nz,ntype)
		#print 'the shape of the CMB values are'
		#print values.shape
		return self.pdf.get_nlp(values)#erased the transpose to the above effect

class BAOPosterior(Posterior):
	"""Posterior constraint on the parallel and perpendicular scale factors from BAO.

	Args:
		name(str): Name to associate with this posterior.
		zpost(double): Redshift where posterior should be evaluated.
		apar(double): Line-of-sight (parallel) scale factor measured using BAO.
		sigma_apar(double): RMS error on measured apar.
		aperp(double): Transverse (perpendicular) scale factor measured using BAO.
		sigma_aperp(double): RMS error on measured aperp.
		rho(double): Correlation coefficient between apar and aperp.
			Must be between -1 and +1.

	Raises:
		AssertionError: The redshift z is not an element of zprior.
	"""
	def __init__(self,name,zpost,apar,sigma_apar,aperp,sigma_aperp,rho,rsdrag):
		self.rsdrag = rsdrag
		self.pdf = GaussianPdf2D(apar,sigma_apar,aperp,sigma_aperp,rho)
		Posterior.__init__(self,name,zpost)

	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		The posterior is applied simultaneously to DH(z)/rs(zd) and DA(z)/rs(zd).

		Args:
			DHz(ndarray): Array of DH(zpost) values to use.
			DAz(ndarray): Array of DA(zpost) values to use.

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		values = np.dstack([DHz,DAz])/self.rsdrag#see CMB posterior comments
		return self.pdf.get_nlp(values)

class SNPosterior_unused(Posterior):
	"""Posterior constraint on mu(z).

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
		mu(float): Central value of mu(z).
		mu_error(float): RMS error on mu(z).
	"""
	def __init__(self,name,zpost,mu,mu_error):
		#print mu
		#mu = np.array([mu])
		#print mu.shape
		#mu = mu[:,np.newaxis]#trying to make all the means of the form (nz,ntype) since the values will be of the form (nsample,nz,ntype)
		self.pdf = GaussianPdf1D(mu,mu_error)
		Posterior.__init__(self,name,zpost)

	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		Args:
			DHz(ndarray): Array of DH(zpost) values to use (will be ignored).
			DAz(ndarray): Array of DA(zpost) values to use.

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		#muz = muz[:,:,np.newaxis]
		return self.pdf.get_nlp(muz)

class SNPosterior(Posterior):
	"""Posterior constraint on mu(z).

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
		mu(float): Central value of mu*(z): actually mu(z)-(M_1=19.05).
		mu_error(float): RMS error on mu(z).
	"""
	def __init__(self,name,zpost,mu,mu_error):
		mean = mu.T
		self.pdf = GaussianPdf(mean,mu_error)
		Posterior.__init__(self,name,zpost)


	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		Args:
			DHz(ndarray): Array of DH(zpost) values to use (will be ignored).
			DAz(ndarray): Array of DA(zpost) values to use (also ignored).
			muz(ndarray): this is no longer the 5log(DL/10pc) but instead 5 log (DL/DH0)

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		muz = muz[:,:,np.newaxis]
		return self.pdf.get_nlp(muz)

class BOSS2016Posterior(Posterior):
    def __init__(self,name,zpost,mean,cov,rsdrag):
        self.rsdrag = rsdrag
        #print 'BOSS mean shape is'
        #print mean.shape
        self.pdf = GaussianPdf(mean,cov)
        Posterior.__init__(self,name,zpost)
    #trying to make all the means of the form (nz,ntype) since the values will be of the form (nsample,ntype,nz)

    def constraint(self,DHz,DAz,muz):
        Hz = 2.99792458e5 /DHz
        values = np.dstack([Hz,DAz])
        #print 'BOSS 2016 val shape is'
        #print values.shape
        #print 'the shape of BOSS values are'
        #print values.shape
        return self.pdf.get_nlp(values)
        
class DESIPosterior(Posterior):
    def __init__(self,name,zpost,mean,cov,rsdrag):
        self.rsdrag = rsdrag
        #print 'DESI mean shape is'
        #print mean.shape
        self.pdf = GaussianPdf(mean,cov)
        Posterior.__init__(self,name,zpost)
    #trying to make all the means of the form (nz,ntype) since the values will be of the form (nsample,ntype,nz)

    def constraint(self,DHz,DAz,muz):
        #Hz = 2.99792458e5 /DHz
        values = np.dstack([DHz,DAz])
        #print 'DESI 2016 val shape is'
        #print values.shape
        #print 'the shape of BOSS values are'
        #print values.shape
        return self.pdf.get_nlp(values)        
