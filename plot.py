#!/usr/bin/env python
"""Plots expansion history inferences.
"""

import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt

import gphist

clight = 299792.458

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',type = str, default = None,
        help = 'name of input file(s) to read (wildcard patterns are supported)')
    parser.add_argument('--posterior', type = str, action='append', metavar = 'NAME',
        help = 'posteriors to plot (can be repeated, plot all if None)')
    parser.add_argument('--full', action = 'store_true',
        help = 'show plots of DH/DH0,DA/DA0 evolution over full redshift range')
    parser.add_argument('--zoom', action = 'store_true',
        help = 'show plots of DH,DA on linear scale up to redshift zmax')
    parser.add_argument('--dark-energy', action = 'store_true',
        help = 'show plots of H(z)/(1+z) and Omega_DE(z)/Omega_DE(0) up to redshift zmax')
    parser.add_argument('--level', type = float, default = 0.9,
        help = 'confidence level to plot')
    parser.add_argument('--no-examples', action = 'store_true',
        help = 'do not show examples of random realizations of DH(z) and DA(z)')
    parser.add_argument('--zmax', type = float, default = 3.0,
        help = 'maximum redshift to display on H(z)/(1+z) plot')
    parser.add_argument('--output', type = str, default = '',
        help = 'base name for saving plots')
    parser.add_argument('--show', action = 'store_true',
        help = 'show each plot')
    parser.add_argument('--plot-format', type = str, default = 'png', metavar = 'FMT',
        help = 'format for saving plots (png,pdf,...)')
    args = parser.parse_args()

    # Do we have any inputs to read?
    if args.input is None:
        print 'Missing required input arg.'
        return -1
    input_files = glob.glob(args.input)
    if not input_files:
        input_files = glob.glob(args.input + '.npz')
    if not input_files:
        print 'No input files match the pattern %r' % args.input
        return -1

    # Do we have anything to plot?
    num_plot_rows = args.full + args.zoom + args.dark_energy
    if num_plot_rows == 0:
        print 'No plots selected.'
        return 0
    show_examples = not args.no_examples

    # Combine the histograms from each input file.
    for index,input_file in enumerate(input_files):
        loaded = np.load(input_file)
        if index == 0:
            DH_hist = loaded['DH_hist']
            DA_hist = loaded['DA_hist']
            DH0 = loaded['DH0']
            DA0 = loaded['DA0']
            zevol = loaded['zevol']
            bin_range = loaded['bin_range']
            if show_examples:
                DH_realizations = loaded['DH_realizations']
                DA_realizations = loaded['DA_realizations']
            posterior_names = loaded['posterior_names']
        else:
            assert np.array_equal(DH0,loaded['DH0']),'Found inconsistent DH0'
            assert np.array_equal(DA0,loaded['DA0']),'Found inconsistent DA0'
            assert np.array_equal(zevol,loaded['zevol']),'Found inconsistent zevol'
            assert np.array_equal(bin_range,loaded['bin_range']),\
                'Found inconsistent bin_range'
            assert np.array_equal(posterior_names,loaded['posterior_names']),\
                'Found inconsistent posterior_names'
            DH_hist += loaded['DH_hist']
            DA_hist += loaded['DA_hist']

    # Loop over posterior permutations.
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
    for iperm,perm in enumerate(perms):

        name = '-'.join(posterior_names[perms[iperm]]) or 'Prior'
        if args.posterior and name not in args.posterior:
            continue

        print '%d : %s' % (iperm,name)

        # Calculate the confidence bands of DH/DH0 and DA/DA0.
        DH_ratio_limits = gphist.analysis.calculate_confidence_limits(
            DH_hist[iperm],[args.level],bin_range)
        DA_ratio_limits = gphist.analysis.calculate_confidence_limits(
            DA_hist[iperm],[args.level],bin_range)

        # Convert to limits on DH, DA.
        DH_limits = DH_ratio_limits*DH0
        DA_limits = DA_ratio_limits*DA0

        # Find first z index beyond H(z)/(1+z) plot.
        iend = 1+np.argmax(zevol > args.zmax)

        fig = plt.figure(name,figsize=(12,4*num_plot_rows))
        fig.subplots_adjust(left=0.06,bottom=0.07,right=0.98,top=0.99,wspace=0.15,hspace=0.18)
        fig.set_facecolor('white')
        irow = 0

        # Plot evolution of DH/DH0, DA/DA0 over full redshift range.
        if args.full:

            plt.subplot(num_plot_rows,2,2*irow+1)
            plt.xscale('log')
            plt.grid(True)
            plt.xlim([1.,1000.])
            plt.ylim([0.5,1.7])
            plt.fill_between(1+zevol,DH_ratio_limits[0],DH_ratio_limits[-1],
                facecolor='blue',alpha=0.25)
            plt.plot(1+zevol,DH_ratio_limits[0],'b--')
            plt.plot(1+zevol,DH_ratio_limits[1],'b-')
            plt.plot(1+zevol,DH_ratio_limits[2],'b--')
            if show_examples:
                plt.plot(1+zevol,(DH_realizations[iperm]/DH0).T,'r',alpha=0.5)
            plt.xlabel(r'$1+z$')
            plt.ylabel(r'$D_H(z)/D_H^0(z)$')

            plt.subplot(num_plot_rows,2,2*irow+2)
            plt.xscale('log')
            plt.grid(True)
            plt.xlim([1.,1000.])
            plt.ylim([0.5,1.7])
            plt.fill_between(1+zevol[1:],DA_ratio_limits[0],DA_ratio_limits[-1],
                facecolor='blue',alpha=0.25)
            plt.plot(1+zevol[1:],DA_ratio_limits[0],'b--')
            plt.plot(1+zevol[1:],DA_ratio_limits[1],'b-')
            plt.plot(1+zevol[1:],DA_ratio_limits[2],'b--')
            if show_examples:
                plt.plot(1+zevol[1:],(DA_realizations[iperm]/DA0).T,'r',alpha=0.5)
            plt.xlabel(r'$1+z$')
            plt.ylabel(r'$D_A(z)/D_A^0(z)$')

            irow += 1

        # Plot zooms of DH,DA up to zmax.
        if args.zoom:

            plt.subplot(num_plot_rows,2,2*irow+1)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zevol[:iend],1e-3*DH_limits[0,:iend],1e-3*DH_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zevol[:iend],1e-3*DH_limits[0,:iend],'b--')
            plt.plot(zevol[:iend],1e-3*DH_limits[1,:iend],'b-')
            plt.plot(zevol[:iend],1e-3*DH_limits[2,:iend],'b--')
            if show_examples:
                plt.plot(zevol[:iend],1e-3*DH_realizations[iperm,:,:iend].T,'r',alpha=0.5)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$D_H(z)$ (Gpc)')

            plt.subplot(num_plot_rows,2,2*irow+2)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zevol[:iend],1e-3*DA_limits[0,:iend],1e-3*DA_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zevol[:iend],1e-3*DA_limits[0,:iend],'b--')
            plt.plot(zevol[:iend],1e-3*DA_limits[1,:iend],'b-')
            plt.plot(zevol[:iend],1e-3*DA_limits[2,:iend],'b--')
            if show_examples:
                plt.plot(zevol[:iend],1e-3*DA_realizations[iperm,:,:iend].T,'r',alpha=0.5)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$D_A(z)$ (Gpc)')

            irow += 1

        # Plot dark-energy diagnostics up to zmax.
        if args.dark_energy:

            # Calculate the corresponding limits and realizations acceleration H(z)/(1+z).
            accel_limits = clight/DH_limits/(1+zevol)
            if show_examples:
                accel_realizations = clight/DH_realizations[iperm]/(1+zevol)

            plt.subplot(num_plot_rows,2,2*irow+1)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zevol[:iend],accel_limits[0,:iend],accel_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zevol[:iend],accel_limits[0,:iend],'b--')
            plt.plot(zevol[:iend],accel_limits[1,:iend],'b-')
            plt.plot(zevol[:iend],accel_limits[2,:iend],'b--')
            if show_examples:
                plt.plot(zevol[:iend],accel_realizations[:,:iend].T,'r',alpha=0.5)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$H(z)/(1+z)$ (Mpc)')

            irow += 1

        plt.savefig(args.output + name + '.' + args.plot_format)
        if args.show:
            plt.show()

if __name__ == '__main__':
    main()
