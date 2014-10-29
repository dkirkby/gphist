#!/usr/bin/env python
"""Plots expansion history inferences.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import gphist

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',type = str, default = None,
        help = 'name of input file to read (extension .npz will be added)')
    parser.add_argument('--posteriors',type=int, default = 0,
        help = 'posteriors permutation index to use')
    parser.add_argument('--save', type = str, default = None,
        help = 'name of plot to save')
    args = parser.parse_args()

    if args.input is None:
        print 'Missing required input arg.'
        return -1

    loaded = np.load(args.input+'.npz')
    DH_hist = loaded['DH_hist']
    DA_hist = loaded['DA_hist']
    DH0 = loaded['DH0']
    DA0 = loaded['DA0']
    zevol = loaded['zevol']
    bin_range = loaded['bin_range']
    posterior_names = loaded['posterior_names']

    npost = len(posterior_names)
    if args.posteriors < 0 or args.posteriors >= 2**npost:
        print 'Invalid posteriors %d (should be in the range 0 - %d)' % (
            args.posteriors,2**npost-1)
        return -1
    perms = gphist.analysis.get_permutations(npost)
    posteriors_name = '+'.join(posterior_names[perms[args.posteriors]]) or 'None'
    print '%d = %s' % (args.posteriors,posteriors_name)

    levels = np.array([0.05,0.5,0.95])
    DH_limits = gphist.analysis.calculate_confidence_limits(
        DH_hist[args.posteriors],levels,bin_range)
    DA_limits = gphist.analysis.calculate_confidence_limits(
        DA_hist[args.posteriors],levels,bin_range)

    fig = plt.figure(figsize=(14,6))
    fig.set_facecolor('white')

    plt.subplot(1,2,1)
    plt.xscale('log')
    plt.grid(True)
    plt.xlim([1.,1000.])
    plt.ylim([0.5,1.7])
    plt.fill_between(1+zevol,DH_limits[0],DH_limits[-1],facecolor='blue',alpha=0.25)
    plt.plot(1+zevol,DH_limits[0],'b--')
    plt.plot(1+zevol,DH_limits[1],'b-')
    plt.plot(1+zevol,DH_limits[2],'b--')
    plt.xlabel('log(1+z)')
    plt.ylabel('DH(z)/DH0(z)')

    plt.subplot(1,2,2)
    plt.xscale('log')
    plt.grid(True)
    plt.xlim([1.,1000.])
    plt.ylim([0.5,1.7])
    plt.fill_between(1+zevol[1:],DA_limits[0],DA_limits[-1],facecolor='blue',alpha=0.25)
    plt.plot(1+zevol[1:],DA_limits[0],'b--')
    plt.plot(1+zevol[1:],DA_limits[1],'b-')
    plt.plot(1+zevol[1:],DA_limits[2],'b--')
    plt.xlabel('log(1+z)')
    plt.ylabel('DA(z)/DA0(z)')

    if args.save:
        plt.savefig(args.save)

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(DH_hist[args.posteriors,0])
    plt.subplot(1,2,2)
    plt.plot(DA_hist[args.posteriors,-1])

    plt.show()

if __name__ == '__main__':
    main()
