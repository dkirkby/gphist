#!/usr/bin/env python
"""Infer the cosmological expansion history using a Gaussian process prior.
"""

import argparse

import numpy as np

import gphist

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type = int, default = 26102014,
        help = 'random seed to use for sampling the prior')
    parser.add_argument('--num-samples', type = int, default = 1000000,
        help = 'number of samples to generate')
    parser.add_argument('--num-evol-hist', type = int, default = 50,
        help = 'number of evolution variable steps to use for histogramming')
    parser.add_argument('--max-array-size', type = float, default = 1.0,
        help = 'maximum array memory allocation size in gigabytes')
    parser.add_argument('--hyper-h', type = float, default = 0.1,
        help = 'vertical scale hyperparameter value to use')
    parser.add_argument('--hyper-sigma', type = float, default = 0.05,
        help = 'horizontal scale hyperparameter value to use')
    parser.add_argument('--hyper-index', type = int, default = None,
        help = 'index into hyperparameter marginalization grid to use (ignore if None)')
    parser.add_argument('--hyper-count', type = int, default = 1,
        help = 'number of consecutive marginalization grid indices to run')
    parser.add_argument('--hyper-num-h', type = int, default = 10,
        help = 'number of h grid points in marginalization grid')
    parser.add_argument('--hyper-h-min', type = float, default = 0.02,
        help = 'minimum value of hyperparameter h for marginalization grid')
    parser.add_argument('--hyper-h-max', type = float, default = 0.5,
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
    parser.add_argument('--rsdrag', type = float, default = 147.36,
        help = 'nominal sound horizon rs(zdrag) at the drag epoch in Mpc')
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

    # Define intermediate redshifts where the prior needs to be sampled.
    zLRG = 0.57
    zLya = 2.3
    z_extra = [zLRG, zLya]

    # Initialize the posteriors to use.
    posteriors = [
        # Local H0 measurement from Reis 2013.
        gphist.posterior.LocalH0Posterior('H0'),
        # BOSS LRG BAO from Anderson 2014.
        gphist.posterior.BAOPosterior('LRG',zLRG,20.74,0.69,14.95,0.21,-0.52,args.rsdrag),
        # BOSS Lya-Lya & QSO-Lya from Delubac 2014.
        gphist.posterior.BAOPosterior('Lya',zLya,9.15,1.22,36.46,0.20,-0.38,args.rsdrag),
        # Extended CMB case from Shahab Nov-4 email.
        gphist.posterior.CMBPosterior('CMB',args.zstar,0.1871433E+00,0.1238882E+02,
            6.57448e-05,0.00461449,0.338313)
    ]
    posterior_names = np.array([p.name for p in posteriors])

    return -1

    # Initialize the evolution variable.
    evol = gphist.evolution.LogScale(args.num_sample_steps,args.zstar,z_extra)

    # Initialize the distance model.
    model = gphist.distance.HubbleDistanceModel(evol)

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

        # Break the calculation into cycles to limit the memory consumption.
        for cycle in range(args.num_cycles):

            # Initialize a unique random state for this cycle in a
            # portable and reproducible way.
            random_state = np.random.RandomState([args.seed,hyper_offset,cycle])

            # Generate samples from the prior using sequential seeds.
            samples = prior.generate_samples(args.num_samples,evol.svalues,random_state)

            # Convert each sample into a corresponding tabulated DH(z).
            DH = model.get_DH(samples)
            del samples

            # Calculate the corresponding comoving distance functions DC(z).
            DC = evol.get_DC(DH)

            # Calculate the corresponding comoving angular scale functions DA(z).
            DA = gphist.distance.convert_DC_to_DA(DH,DC,args.omega_k)

            # Calculate -logL for each combination of posterior and prior sample.
            posteriors_nll = gphist.analysis.calculate_posteriors_nll(DH,DA,posteriors)

            # Select some random realizations for each combination of posteriors.
            # Note that when sigma < 1/num_sample_steps, the realizations cannot be
            # reconstructed by interpolation.
            # For now, we just sample the first cycle but it might be better to sample
            # all cycles and then downsample.
            if cycle == 0:
                DH_realizations,DA_realizations = gphist.analysis.select_random_realizations(
                    DH,DA,posteriors_nll,args.num_save)

            # Downsample for histogramming. Note that we use DC0 for DA0, i.e., assuming
            # zero curvature for the baseline.
            z_ds,DH_ds,DA_ds,DH0_ds,DA0_ds = gphist.analysis.downsample(
                args.num_hist_steps,evol.zvalues,DH,DA,model.DH0,model.DC0)

            # Build histograms of DH/DH0 and DA/DA0 for each redshift slice and
            # all permutations of posteriors.
            DH_hist,DA_hist = gphist.analysis.calculate_distance_histograms(
                DH_ds,DH0_ds,DA_ds,DA0_ds,posteriors_nll,
                args.num_bins,args.min_ratio,args.max_ratio)

            # Combine with the results of any previous cycles.
            if cycle == 0:
                combined_DH_hist = DH_hist
                combined_DA_hist = DA_hist
            else:
                combined_DH_hist += DH_hist
                combined_DA_hist += DA_hist            

            print 'Finished cycle %d of %d' % (cycle+1,args.num_cycles)

        # Save the combined results for these hyperparameters.
        if args.output:
            fixed_options = np.array([args.num_samples,args.num_cycles,
                args.hyper_num_h,args.hyper_num_sigma])
            variable_options = np.array([args.seed,hyper_index,hyper_offset])
            bin_range = np.array([args.min_ratio,args.max_ratio])
            hyper_range = np.array([args.hyper_h_min,args.hyper_h_max,
                args.hyper_sigma_min,args.hyper_sigma_max])
            output_name = '%s.%d.npz' % (args.output,hyper_offset)
            np.savez(output_name,
                DH_hist=combined_DH_hist,DA_hist=combined_DA_hist,
                DH0=DH0_ds,DA0=DA0_ds,zhist=z_ds,zevol=evol.zvalues,
                DH0_full=model.DH0,DA0_full=model.DC0,
                fixed_options=fixed_options,variable_options=variable_options,
                bin_range=bin_range,hyper_range=hyper_range,
                DH_realizations=DH_realizations,DA_realizations=DA_realizations,
                posterior_names=posterior_names)
            print 'Wrote %s' % output_name

if __name__ == '__main__':
    main()
