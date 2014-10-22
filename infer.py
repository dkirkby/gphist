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
    parser.add_argument('--hyper-h', type = float, default = 0.3,
        help = 'vertical scale hyperparameter value to use')
    parser.add_argument('--hyper-sigma', type = float, default = 1.0,
        help = 'horizontal scale hyperparameter value to use')
    args = parser.parse_args()

    # Initialize the Gaussian process prior.
    gp = gphist.process.SquaredExponentialGaussianProcess(args.hyper_h,args.hyper_sigma)

    # Initialize the evolution variable.
    slist = np.linspace(0.,10.,11)

    # Generate samples...
    gamma = gp.sample(args.num_samples,slist)
    print gamma

if __name__ == '__main__':
    main()
