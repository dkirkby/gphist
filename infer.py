#!/usr/bin/env python
"""Infers the expansion history for a fixed set of assumptions.
"""

import argparse

import numpy as np

import gphist

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-samples', type = int, default = 1000000,
        help = 'number of samples to generate')
    parser.add_argument('--num-steps', type = int, default = 50,
        help = 'number of steps in evolution variable to use')
    parser.add_argument('--hyper-h', type = float, default = 0.3,
        help = 'vertical scale hyperparameter value to use')
    parser.add_argument('--hyper-sigma', type = float, default = 0.14295333304236352,
        help = 'horizontal scale hyperparameter value to use')
    parser.add_argument('--omega-k', type = float, default =0.,
        help = 'curvature parameter')
    parser.add_argument('--zstar', type = float, default = 1090.48,
        help = 'redshift of last scattering')
    parser.add_argument('--num-bins', type = int, default = 2000,
        help = 'number of bins to use for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--min-ratio', type = float, default = 0.,
        help = 'minimum ratio for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--max-ratio', type = float, default = 2.,
        help = 'maximum ratio for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--num-save', type = int, default = 5,
        help = 'number of prior realizations to save for each combination of posteriors')
    parser.add_argument('--output', type = str, default = None,
        help = 'name of output file to write (the extension .npz will be added)')
    args = parser.parse_args()

    # Initialize the Gaussian process prior.
    prior = gphist.process.SquaredExponentialGaussianProcess(args.hyper_h,args.hyper_sigma)

    # Initialize the evolution variable.
    evol = gphist.evolution.LogScale(args.num_steps,args.zstar)

    # Initialize the distance model.
    model = gphist.distance.HubbleDistanceModel(evol)

    # Generate samples from the prior.
    samples = prior.generate_samples(args.num_samples,evol.svalues)

    # Convert each sample into a corresponding tabulated DH(z).
    DH = model.get_DH(samples)

    # Calculate the corresponding comoving distance functions DC(z).
    DC = evol.get_DC(DH)

    # Calculate the corresponding comoving angular scale functions DA(z).
    DA = gphist.distance.convert_DC_to_DA(DH,DC,args.omega_k)

    # Initialize the posteriors to use.
    posteriors = [
        gphist.posterior.CMBPosterior(),
        gphist.posterior.LocalH0Posterior(),
    ]

    # Calculate -logL for each combination of posterior and prior sample.
    posteriors_nll = gphist.analysis.calculate_posteriors_nll(DH,DA,posteriors)
    print np.max(posteriors_nll,axis=1)

    # Build histograms of DH/DH0 and DA/DC0 for each redshift slice and
    # all permutations of posteriors.
    bin_range = np.array([args.min_ratio,args.max_ratio])
    DH_hist,DA_hist = gphist.analysis.calculate_distance_histograms(
        DH,model.DH0,DA,model.DC0,posteriors_nll,args.num_bins,bin_range)

    # Select some random realizations for each combination of posteriors.
    # ...

    # Save outputs.
    if args.output:
        np.savez(args.output+'.npz',DH_hist=DH_hist,DA_hist=DA_hist,
            zevol=evol.zvalues,bin_range=bin_range)

if __name__ == '__main__':
    main()
