#!/usr/bin/env python
"""Infer the cosmological expansion history using a Gaussian process prior.
"""

import argparse
import math

import numpy as np

import gphist
import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type = int, default = 26102014,
        help = 'random seed to use for sampling the prior')
    parser.add_argument('--num-samples', type = int, default = 1000000,
        help = 'number of samples to generate')
    parser.add_argument('--num-evol-hist', type = int, default = 100,
        help = 'number of equally spaced evolution variable steps to use for histogramming')
    parser.add_argument('--max-array-size', type = float, default = 1.0,
        help = 'maximum array memory allocation size in gigabytes')
    parser.add_argument('--hyper-h', type = float, default = 0.1,
        help = 'vertical scale hyperparameter value to use')
    parser.add_argument('--hyper-sigma', type = float, default = 0.02,
        help = 'horizontal scale hyperparameter value to use')
    parser.add_argument('--hyper-index', type = int, default = None,
        help = 'index into hyperparameter marginalization grid to use (ignore if None)')
    parser.add_argument('--hyper-count', type = int, default = 1,
        help = 'number of consecutive marginalization grid indices to run')
    parser.add_argument('--hyper-num-h', type = int, default = 10,
        help = 'number of h grid points in marginalization grid')
    parser.add_argument('--hyper-h-min', type = float, default = 0.01,
        help = 'minimum value of hyperparameter h for marginalization grid')
    parser.add_argument('--hyper-h-max', type = float, default = 1.0,
        help = 'maximum value of hyperparameter h for marginalization grid')
    parser.add_argument('--hyper-num-sigma', type = int, default = 20,
        help = 'number of sigma grid points in marginalization grid')
    parser.add_argument('--hyper-sigma-min', type = float, default = 0.001,
        help = 'minimum value of hyperparameter sigma for marginalization grid')
    parser.add_argument('--hyper-sigma-max', type = float, default = 1.0,
        help = 'maximum value of hyperparameter sigma for marginalization grid')
    parser.add_argument('--omega-k', type = float, default =0.,
        help = 'curvature parameter')
    parser.add_argument('--zstar', type = float, default = 1090.48,
        help = 'nominal redshift of last scattering')
    parser.add_argument('--zLRG', type = float, default = 0.57,
        help = 'redshift of LRG BAO constraint')
    parser.add_argument('--zLya', type = float, default = 2.3,
        help = 'redshift of Lyman-alpha BAO constraint')
    parser.add_argument('--rsdrag', type = float, default = 147.36,
        help = 'nominal sound horizon rs(zdrag) at the drag epoch in Mpc')
    parser.add_argument('--dark-energy', action= 'store_true',
        help = 'calculate dark energy expansion history for each realization')
    parser.add_argument('--growth', action= 'store_true',
        help = 'calculate growth functions for each realization')
    parser.add_argument('--accel', action= 'store_true',
        help = 'calculate deceleration parameter for each realization') 
    parser.add_argument('--forecast', action= 'store_true',
        help = 'adds a constraint for forecasting DESI results')
    parser.add_argument('--Neff', action= 'store_true',
        help = 'use distances for case where Neff varies')                
    parser.add_argument('--num-bins', type = int, default = 1000,
        help = 'number of bins to use for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--min-ratio', type = float, default = 0.5,
        help = 'minimum ratio for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--max-ratio', type = float, default = 1.5,
        help = 'maximum ratio for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--num-save', type = int, default = 5,
        help = 'number of prior realizations to save for each combination of posteriors')
    parser.add_argument('--output', type = str, default = None,
        help = 'name of output file to write (the extension .npz will be added)')
    args = parser.parse_args()


    
    SN_data = np.loadtxt('jla_mub.txt')	
    z_SN = SN_data[:,0]
    mu_SN = SN_data[:,1]

    SN_cov = np.loadtxt('jla_mub_covmatrix.dat')
    cov_mu = np.zeros((len(mu_SN),len(mu_SN)))
    for i in range(len(SN_cov)-1):
        cov_mu[i/31,i%31] = SN_cov[i+1]
   
	

    BOSS_cov_z1 = np.loadtxt('BAO_BEUTLER_cov_z1.txt')
    BOSS_cov_z2 = np.loadtxt('BAO_BEUTLER_cov_z2.txt')
    BOSS_cov_z3 = np.loadtxt('BAO_BEUTLER_cov_z3.txt')
    cov_BOSS = np.dstack([BOSS_cov_z1,BOSS_cov_z2,BOSS_cov_z3])
    BOSS_data = np.loadtxt('BAO_BEUTLER_results_nostrings.txt')
    z_BOSS = BOSS_data[:,0]
    dist_BOSS = BOSS_data[:,1:]

    DESI_data = np.loadtxt('DESI_means.txt')
    z_DESI = DESI_data[0]
    DA_DESI = DESI_data[2]
    DH_DESI = DESI_data[1]
    dist_DESI = np.dstack([DH_DESI,DA_DESI])[0]
    DESI_cov = np.loadtxt('DESI_cov.txt')
    DA_cov = DESI_cov[:,0]*0.01
    DH_cov = DESI_cov[:,1]*0.01
    cov_DESI = np.zeros((2,2,len(z_DESI)))
    cov_DESI[0,0,:] = (DA_DESI*DA_cov)**2
    cov_DESI[1,1,:] = (DH_DESI*DH_cov)**2
    cov_DESI[0,1,:] = DH_DESI*DA_DESI*DA_cov*DH_cov*-0.38
    cov_DESI[1,0,:] = cov_DESI[0,1,:]    

    # Initialize the posteriors to use.
    posteriors = [
        # Debugging posteriors: 0.1% measurements of DH,DA at z=2.
        #gphist.posterior.DHPosterior('DH',2.0,1450.0,1.45),
        #phist.posterior.DAPosterior('DA',2.0,5300.0,5.3),
        #gphist.posterior.BAOPosterior('DH+DA',2.0,
        #    1450.0/args.rsdrag,1.45/args.rsdrag,
        #    5300.0/args.rsdrag,5.3/args.rsdrag,0.,args.rsdrag),

        # Local H0 measurement from Reis 2011 (http://dx.doi.org/10.1088/0004-637X/730/2/119)
        gphist.posterior.LocalH0Posterior('H0',73.24,1.74),

        # BOSS LRG BAO from Anderson 2014.
        #gphist.posterior.BAOPosterior('LRG',args.zLRG,20.74,0.69,14.95,0.21,-0.52,args.rsdrag),
        #gphist.posterior.BOSS2016Po

        # BOSS Lya-Lya & QSO-Lya from Delubac 2014.
        #gphist.posterior.BAOPosterior('Lya',args.zLya,9.15,0.20,36.46,1.22,-0.38,args.rsdrag),
        #from Bautista 2017
        #gphist.posterior.BAOPosterior('Lya',args.zLya,9.07,0.31,37.77,2.13,-0.38,args.rsdrag),
        #Bautista data but rs is allowed to vary
        gphist.posterior.ScalePosteriorLya('Lya',args.zLya,9.07,0.31,37.77,2.13,-0.38,args.rsdrag),
        #gphist.posterior.DHPosterior('LyaDH',args.zLya,9.15*args.rsdrag,0.20*args.rsdrag),
	
	# SN posterior
	#gphist.posterior.SNPosterior('SN',z_SN[0],mu_SN[0],cov_mu[0,0]),
    ]

    # The choice of CMB posterior depends on whether we are inferring the dark-energy evolution.
    if args.Neff:
        print 'LCDM+Neff:TT'
        #posteriors.append(
        #    gphist.posterior.CMBPosterior('CMB',args.zstar,0.191908,12.727515,
        #    1.56e-06,5.861e-05,0.00241565))
        posteriors.append(
            gphist.posterior.CMBPosterior('CMB',args.zstar,1.8636260E-01,1.215866E+01,
            3.51315078E-05,3.1234742E-03,2.823618E-01))
            
    else:
        print 'LCDM:TT'
        posteriors.append(
            gphist.posterior.CMBPosterior('CMB',args.zstar,1.9273724E-01,1.2749623E+01,
            1.68889856E-06,5.894375E-05,2.2449463E-03))


    SN_post = gphist.posterior.SNPosterior('SN',z_SN,np.array([mu_SN]),cov_mu)
    #BOSS2016post = gphist.posterior.BOSS2016Posterior('LRG', z_BOSS, dist_BOSS ,cov_BOSS,args.rsdrag)
    BOSS2016post = gphist.posterior.ScalePosteriorLRG('LRG', z_BOSS, dist_BOSS ,cov_BOSS,args.rsdrag)
       	
    posterior_names = np.array([p.name for p in posteriors])
    posterior_redshifts = np.array([p.zpost for p in posteriors])

    posterior_redshifts = np.concatenate((posterior_redshifts,SN_post.zpost))
    posterior_redshifts = np.concatenate((posterior_redshifts,BOSS2016post.zpost))

    posterior_names = np.concatenate((posterior_names,np.array([SN_post.name])))
    posterior_names = np.concatenate((posterior_names,np.array([BOSS2016post.name])))

    posteriors.append(SN_post)
    posteriors.append(BOSS2016post)
    
    if args.forecast:
        DESIpost = gphist.posterior.DESIPosterior('DESI',z_DESI,dist_DESI,cov_DESI,args.rsdrag)
        posterior_redshifts = np.concatenate((posterior_redshifts,DESIpost.zpost))
        posterior_names = np.concatenate((posterior_names,np.array([DESIpost.name])))
        posteriors.append(DESIpost)

    # Initialize a grid of hyperparameters, if requested.
    if args.hyper_index is not None:
        hyper_grid = gphist.process.HyperParameterLogGrid(
            args.hyper_num_h,args.hyper_h_min,args.hyper_h_max,
            args.hyper_num_sigma,args.hyper_sigma_min,args.hyper_sigma_max)
    else:
        hyper_grid = None

    # Loop over hyperparameter values.
    for hyper_offset in range(args.hyper_count):

        if hyper_grid:
            hyper_index = args.hyper_index + hyper_offset
            h,sigma = hyper_grid.get_values(hyper_index)
        else:
            hyper_index = None
            h,sigma = args.hyper_h,args.hyper_sigma

        print 'Using hyperparameters (h,sigma) = (%f,%f)' % (h,sigma)

        # Initialize the Gaussian process prior.
        prior = gphist.process.SquaredExponentialGaussianProcess(h,sigma)

        # Calculate the amount of oversampling required in the evolution variable to
        # sample the prior given this value of sigma.
        min_num_evol = math.ceil(2./sigma)
        num_evol,evol_oversampling,samples_per_cycle = gphist.evolution.initialize(
            min_num_evol,args.num_evol_hist,args.num_samples,args.max_array_size)

        print 'Using %dx oversampling and %d cycles of %d samples/cycle.' % (
            evol_oversampling,math.ceil(1.*args.num_samples/samples_per_cycle),
            samples_per_cycle)

        # Initialize the evolution variable.
        evol = gphist.evolution.LogScale(num_evol,evol_oversampling,posterior_redshifts)

        # Initialize the distance model.
        model = gphist.distance.HubbleDistanceModel(evol)
        DH0 = model.DH0
        DA0 = model.DC0 # assuming zero curvature


        # Initialize a reproducible random state for this offset. We use independent
        # states for sampling the prior and selecting random realizations so that these
        # are independently reproducible.
        sampling_random_state = np.random.RandomState([args.seed,hyper_offset])
        realization_random_state = np.random.RandomState([args.seed,hyper_offset])

        # Break the calculation into cycles to limit the memory consumption.
        combined_DH_hist,combined_DA_hist,combined_de_hist = None,None,None
        samples_remaining = args.num_samples
        while samples_remaining > 0:

            samples_per_cycle = min(samples_per_cycle,samples_remaining)
            samples_remaining -= samples_per_cycle

            # Generate samples from the prior.
            samples = prior.generate_samples(samples_per_cycle,evol.svalues,
                sampling_random_state)

            # Convert each sample into a corresponding tabulated DH(z).
            DH = model.get_DH(samples)
            del samples
            # Free the large sample array before allocating a large array for DC.
            
            # Calculate the corresponding comoving distance functions DC(z).
            DC = evol.get_DC(DH)

            # Calculate the corresponding comoving angular scale functions DA(z).
            DA = gphist.distance.convert_DC_to_DA(DH,DC,args.omega_k)
            mu = evol.get_mu(DH,DC,evol.zvalues)
            apar = evol.get_apar(DH,DH0[-1],args.rsdrag,evol.zvalues)
            aperp = evol.get_aperp(DH,DA,DH0[-1],args.rsdrag,evol.zvalues)

            # Calculate -logL for each combination of posterior and prior sample.
            posteriors_nlp = gphist.analysis.calculate_posteriors_nlp(
                evol.zvalues,DH,DA,mu,apar,aperp,posteriors)

            # Select some random realizations for each combination of posteriors.
            # For now, we just sample the first cycle but it might be better to sample
            # all cycles and then downsample.
            if combined_DH_hist is None:
                DH_realizations,DA_realizations = gphist.analysis.select_random_realizations(
                    DH,DA,posteriors_nlp,args.num_save,realization_random_state)

            # Downsample distance functions in preparation for histogramming.
            i_ds = evol.downsampled_indices
            z_ds,DH0_ds,DA0_ds = evol.zvalues[i_ds],DH0[i_ds],DA0[i_ds]
            DH_ds,DA_ds = DH[:,i_ds],DA[:,i_ds]
            if args.growth:
                print 'calculating growth functions...'
                phi0_ds,f0_ds = evol.get_phi_take2(DH0_ds[np.newaxis,:],evol.svalues[i_ds])
                phi_ds,f_ds = evol.get_phi_take2(DH_ds,evol.svalues[i_ds])
                print 'done calculating growth functions'
            else:
                phi0_ds,f0_ds = np.ones(DH0_ds.shape),np.ones(DH0_ds.shape)            
                phi_ds,f_ds = np.ones(DH_ds.shape),np.ones(DH_ds.shape)
            
            if args.accel:
                print 'calculating q'
                q = evol.get_accel(DH,evol.svalues)
                q0 = evol.get_accel(DH0[np.newaxis,:],evol.svalues)
                q_ds = q[:,i_ds]
                q0_ds = q0[0,i_ds]
                print 'done calculating q'
            else:
                q_ds = np.ones(DH_ds.shape)
                q0_ds = np.ones(DH0_ds.shape)   

            # Calculate dark energy evolution on the downsampled grid, if requested.
            de0_evol = gphist.cosmology.get_dark_energy_evolution(z_ds,DH0_ds)
            #print args.dark_energy
            if args.dark_energy:
                #print 'dark energy evolution'
                de_evol = gphist.cosmology.get_dark_energy_evolution(z_ds,DH_ds)
            else:
                de_evol = None

            # Build histograms for each downsampled redshift slice and for
            # all permutations of posteriors.
            DH_hist,DA_hist,f_hist,phi_hist,de_hist,q_hist = gphist.analysis.calculate_histograms(
                DH_ds,DH0_ds,DA_ds,DA0_ds,f_ds,f0_ds,phi_ds,phi0_ds,de_evol,de0_evol,q_ds,q0_ds,posteriors_nlp,
                args.num_bins,args.min_ratio,args.max_ratio)

            # Combine with the results of any previous cycles.
            if combined_DH_hist is None:
                combined_DH_hist = DH_hist
                combined_DA_hist = DA_hist
                if args.growth:
                    combined_phi_hist = phi_hist
                    combined_f_hist = f_hist
                if args.accel:    
                    combined_q_hist = q_hist
                if args.dark_energy:
                    combined_de_hist = de_hist
            else:
                combined_DH_hist += DH_hist
                combined_DA_hist += DA_hist
                if args.growth:
                    combined_phi_hist += phi_hist
                    combined_f_hist += f_hist
                if args.accel:    
                    combined_q_hist += q_hist
                if args.dark_energy:
                    combined_de_hist += de_hist

            print 'Finished cycle with %5.2f%% samples remaining.' % (
                100.*samples_remaining/args.num_samples)

        # Save the combined results for these hyperparameters.
        if args.output:
            fixed_options = np.array([args.num_samples,
                args.hyper_num_h,args.hyper_num_sigma])
            variable_options = np.array([args.seed,hyper_index,hyper_offset])
            bin_range = np.array([args.min_ratio,args.max_ratio])
            hyper_range = np.array([args.hyper_h_min,args.hyper_h_max,
                args.hyper_sigma_min,args.hyper_sigma_max])
            output_name = '%s.%d.npz' % (args.output,hyper_offset)
            np.savez(output_name,
                zvalues=z_ds,DH_hist=combined_DH_hist,DA_hist=combined_DA_hist,phi_hist=combined_phi_hist,
                f_hist=combined_f_hist,de_hist=combined_de_hist,q_hist=combined_q_hist,
                DH0=DH0_ds,DA0=DA0_ds,phi0=phi0_ds,f0=f0_ds,de0=de0_evol,q0=q0_ds,
                fixed_options=fixed_options,variable_options=variable_options,
                bin_range=bin_range,hyper_range=hyper_range,
                DH_realizations=DH_realizations,DA_realizations=DA_realizations,
                posterior_names=posterior_names)
            print 'Wrote %s' % output_name

if __name__ == '__main__':
    main()
