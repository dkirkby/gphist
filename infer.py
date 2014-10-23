#!/usr/bin/env python
"""Infers the expansion history for a fixed set of assumptions.
"""

import argparse

import numpy as np

import gphist

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-samples', type = int, default = 1000,
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
    post = gphist.posterior.LocalH0Posterior()
    nll = post.get_nll(DH,DA)
    wgt = np.exp(-nll)

    q = np.array((0.05,0.5,0.95))
    DHq = gphist.analysis.get_quantiles(DH,q,weights=wgt)/model.DH0
    print DHq
    #DAq = gphist.analysis.get_quantiles(DA[:,1:],q)/model.DC0[1:]
    #print DAq

if __name__ == '__main__':
    main()
