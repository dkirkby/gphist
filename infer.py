#!/usr/bin/env python
"""Infers the expansion history for a fixed set of assumptions.
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
    parser.add_argument('--num-steps', type = int, default = 50,
        help = 'number of steps in evolution variable to use')
    parser.add_argument('--num-cycles', type = int, default = 1,
        help = 'number of generation cycles to perform')
    parser.add_argument('--hyper-h', type = float, default = 0.3,
        help = 'vertical scale hyperparameter value to use')
    parser.add_argument('--hyper-sigma', type = float, default = 0.14,
        help = 'horizontal scale hyperparameter value to use')
    parser.add_argument('--omega-k', type = float, default =0.,
        help = 'curvature parameter')
    parser.add_argument('--zstar', type = float, default = 1090.48,
        help = 'nominal redshift of last scattering')
    parser.add_argument('--rsdrag', type = float, default = 147.36,
        help = 'nominal sound horizon rs(zdrag) at the drag epoch in Mpc')
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
    parser.add_argument('--debug', action = 'store_true',
        help = 'use special priors to debug DH,DA,BAO constraints')
    args = parser.parse_args()

    # Initialize the Gaussian process prior.
    prior = gphist.process.SquaredExponentialGaussianProcess(args.hyper_h,args.hyper_sigma)

    # Initialize the evolution variable.
    evol = gphist.evolution.LogScale(args.num_steps,args.zstar)

    # Initialize the distance model.
    model = gphist.distance.HubbleDistanceModel(evol)

    # Initialize the posteriors to use.
    if args.debug:
        # Compare independent DH,DA constraints at z ~ 2.1 with a simultaneous BAO constraint.
        iz = 8
        zref = evol.zvalues[iz]
        print 'Debug: z,DH(z),DC(z) =',zref,model.DH0[iz],model.DC0[iz-1]
        frac = 0.01 # fractional error of constraints
        rho = 0.9 # DH-DC correlation for 2D constraint
        posteriors = [
            gphist.posterior.DHPosterior('DH',evol,zref,model.DH0[iz],frac*model.DH0[iz]),
            gphist.posterior.DAPosterior('DA',evol,zref,model.DC0[iz-1],frac*model.DC0[iz-1]),
            gphist.posterior.BAOPosterior('BAO',evol,zref,
                model.DH0[iz]/args.rsdrag,frac*model.DH0[iz]/args.rsdrag,
                model.DC0[iz-1]/args.rsdrag,frac*model.DC0[iz-1]/args.rsdrag,rho,args.rsdrag)
        ]
    else:
        posteriors = [
            gphist.posterior.LocalH0Posterior('H0'),
            gphist.posterior.BAOPosterior('LRG',evol,0.57,20.74,0.69,14.95,0.21,-0.52,args.rsdrag),
            gphist.posterior.BAOPosterior('Lya',evol,2.3,9.15,1.22,36.46,0.20,-0.38,args.rsdrag),
            #gphist.posterior.CMBPosterior('CMB',evol,0.1921764,0.1274139e2,
            #    2.2012293e-06,7.87634e-05,0.0030466538),
            gphist.posterior.CMBPosterior('CMB',evol,0.1835618,0.1204209e2,
                6.0964331e-05,0.00465422,0.36262126),
        ]
    posterior_names = np.array([p.name for p in posteriors])

    for cycle in range(args.num_cycles):

        # Initialize the random state for this (seed,cycle) combination in a
        # portable and reproducible way.
        random_state = np.random.RandomState([args.seed,cycle])

        # Generate samples from the prior using sequential seeds.
        samples = prior.generate_samples(args.num_samples,evol.svalues,random_state)

        # Convert each sample into a corresponding tabulated DH(z).
        DH = model.get_DH(samples)

        # Calculate the corresponding comoving distance functions DC(z).
        DC = evol.get_DC(DH)

        # Calculate the corresponding comoving angular scale functions DA(z).
        DA = gphist.distance.convert_DC_to_DA(DH,DC,args.omega_k)

        # Calculate -logL for each combination of posterior and prior sample.
        posteriors_nll = gphist.analysis.calculate_posteriors_nll(DH,DA,posteriors)

        # Select some random realizations for each combination of posteriors.
        DH_realizations,DA_realizations = gphist.analysis.select_random_realizations(
            DH,DA,posteriors_nll,args.num_save)

        # Build histograms of DH/DH0 and DA/DA0 for each redshift slice and
        # all permutations of posteriors. Note that we use DC0 for DA0, i.e., assuming
        # zero curvature for the baseline. A side effect of this call is that the
        # DH,DA arrays will be overwritten with the ratios DH/DH0, DA/DA0 (to avoid
        # allocating additional large arrays).
        DH_hist,DA_hist = gphist.analysis.calculate_distance_histograms(
            DH,model.DH0,DA,model.DC0,posteriors_nll,args.num_bins,args.min_ratio,args.max_ratio)

        # Save outputs.
        if args.output:
            bin_range = np.array([args.min_ratio,args.max_ratio])
            output_name = '%s.%d.npz' % (args.output,cycle)
            np.savez(output_name,DH_hist=DH_hist,DA_hist=DA_hist,
                DH0=model.DH0,DA0=model.DC0,zevol=evol.zvalues,bin_range=bin_range,
                DH_realizations=DH_realizations,DA_realizations=DA_realizations,
                posterior_names=posterior_names)
            print 'wrote',output_name

if __name__ == '__main__':
    main()
