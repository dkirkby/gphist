#!/usr/bin/env python
"""Plots expansion history inferences.
"""

import argparse
import glob

import numpy as np
# matplotlib is imported inside main()

import gphist

clight = 299792.458 # speed of light in km/s

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',type = str, default = None,
        help = 'name of input file(s) to read (wildcard patterns are supported)')
    parser.add_argument('--posterior', type = str, action='append', metavar = 'NAME',
        help = 'posteriors to plot (can be repeated, plot all if None)')
    parser.add_argument('--nlp', action = 'store_true',
        help = 'show plots of posterior -logL marginalized over hyperparameters')
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
    if num_plot_rows == 0 and not args.nlp:
        print 'No plots selected.'
        return 0
    show_examples = not args.no_examples

    # Initialize matplotlib.
    import matplotlib as mpl
    if not args.show:
        # Use the default backend, which does not require X11 on unix systems.
        mpl.use('Agg')
    import matplotlib.pyplot as plt

    # Combine the histograms from each input file.
    random_states = { }
    for index,input_file in enumerate(input_files):
        loaded = np.load(input_file)
        if index == 0:
            DH_hist = loaded['DH_hist']
            DA_hist = loaded['DA_hist']
            DH0 = loaded['DH0']
            DA0 = loaded['DA0']
            zvalues = loaded['zvalues']
            fixed_options = loaded['fixed_options']
            bin_range = loaded['bin_range']
            hyper_range = loaded['hyper_range']
            if show_examples:
                DH_realizations = loaded['DH_realizations']
                DA_realizations = loaded['DA_realizations']
            posterior_names = loaded['posterior_names']
            # Initialize the posterior permutations.
            npost = len(posterior_names)
            perms = gphist.analysis.get_permutations(npost)
            # Initialize the hyperparameter grid.
            n_samples,n_h,n_sigma = fixed_options
            h_min,h_max,sigma_min,sigma_max = hyper_range
            hyper_grid = gphist.process.HyperParameterLogGrid(
                n_h,h_min,h_max,n_sigma,sigma_min,sigma_max)
            # Initialize array of marginalized posterior NLP values over hyperparameters.
            hyper_nlp = np.zeros((2**npost,n_h,n_sigma))
            nlp_const = -np.log(n_samples)
            nlp_levels = gphist.analysis.get_delta_chisq(num_dof=2)
        else:
            # Distance arrays might differ by roundoff errors because of different downsampling.
            assert np.array_allclose(DH0,loaded['DH0']),'Found inconsistent DH0'
            assert np.array_allclose(DA0,loaded['DA0']),'Found inconsistent DA0'
            assert np.array_allclose(zvalues,loaded['zvalues']),'Found inconsistent zvalues'
            # The following arrays should be identical.
            assert np.array_equal(bin_range,loaded['bin_range']),\
                'Found inconsistent bin_range'
            assert np.array_equal(posterior_names,loaded['posterior_names']),\
                'Found inconsistent posterior_names'
            assert np.array_equal(fixed_options,loaded['fixed_options']),\
                'Found inconsistent fixed options'
            assert np.array_equal(hyper_range,loaded['hyper_range']),\
                'Found inconsistent hyperparameter grids'
            DH_hist += loaded['DH_hist']
            DA_hist += loaded['DA_hist']
        
        # Always load these arrays.
        zvalues_full = loaded['zvalues_full']
        DH0_full = loaded['DH0_full']
        DA0_full = loaded['DA0_full']
        variable_options = loaded['variable_options']

        seed,hyper_index,hyper_offset = variable_options

        # Check that each input was calculated using a different random state.
        random_state = (seed,hyper_offset)
        if random_state in random_states:
            print 'ERROR: random state %r is duplicated in %s' % (random_state,input_file)
            return -1
        random_states[random_state] = input_file

        # Accumulate marginalized hyperparameter statistics.
        if hyper_index is not None:
            i_h,i_sigma = hyper_grid.decode_index(hyper_index)
            # Calculate the posterior weight of this permutation marginalized over the prior
            # as the sum of histogram weights.  All DH and DA histograms have the same sum
            # of weights so we arbitrarily use the first DH histogram.
            marginal_weights = np.sum(loaded['DH_hist'][:,0,:],axis=1)
            hyper_nlp[:,i_h,i_sigma] += -np.log(marginal_weights) - nlp_const

    # Loop over posterior permutations.
    for iperm,perm in enumerate(perms):

        name = '-'.join(posterior_names[perms[iperm]]) or 'Prior'
        if args.posterior and name not in args.posterior:
            continue
        print '%d : %s' % (iperm,name)

        if args.nlp:
            fig = plt.figure('NLP-'+name,figsize=(10,10))
            fig.set_facecolor('white')
            plt.xscale('linear')
            plt.yscale('log')
            nlp = hyper_nlp[iperm] - np.min(hyper_nlp[iperm])
            plt.pcolormesh(hyper_grid.sigma_edges,hyper_grid.h_edges,nlp,
                cmap='rainbow',rasterized=True)
            if iperm > 0: # Cannot plot contours for the flat prior.
                plt.contour(hyper_grid.sigma,hyper_grid.h,nlp,colors='w',
                    levels=nlp_levels,linestyles=('-','--',':'))
            plt.xlabel(r'Hyperparameter $\sigma$')
            plt.ylabel(r'Hyperparameter $h$')
            plt.savefig(args.output + 'NLP-' + name + '.' + args.plot_format)
            if args.show:
                plt.show()
            plt.close()

        if num_plot_rows > 0:

            # Calculate the confidence bands of DH/DH0 and DA/DA0.
            DH_ratio_limits = gphist.analysis.calculate_confidence_limits(
                DH_hist[iperm],[args.level],bin_range)
            DA_ratio_limits = gphist.analysis.calculate_confidence_limits(
                DA_hist[iperm],[args.level],bin_range)

            # Convert to limits on DH, DA, with DA limits extended to z=0.
            DH_limits = DH_ratio_limits*DH0
            DA_limits = np.empty_like(DH_limits)
            DA_limits[:,1:] = DA_ratio_limits*DA0[1:]
            DA_limits[:,0] = 0.

            # Find first z index beyond zmax.
            iend = 1+np.argmax(zvalues > args.zmax)
            iend_full = 1+np.argmax(zvalues_full > args.zmax)

            fig = plt.figure(name,figsize=(12,4*num_plot_rows))
            fig.subplots_adjust(left=0.06,bottom=0.07,right=0.98,
                top=0.99,wspace=0.15,hspace=0.18)
            fig.set_facecolor('white')
            irow = 0

        # Plot evolution of DH/DH0, DA/DA0 over full redshift range.
        if args.full:

            plt.subplot(num_plot_rows,2,2*irow+1)
            plt.xscale('log')
            plt.grid(True)
            plt.xlim([1.,1+np.max(zvalues)])
            plt.fill_between(1+zvalues,DH_ratio_limits[0],DH_ratio_limits[-1],
                facecolor='blue',alpha=0.25)
            plt.plot(1+zvalues,DH_ratio_limits[0],'b--')
            plt.plot(1+zvalues,DH_ratio_limits[1],'b-')
            plt.plot(1+zvalues,DH_ratio_limits[2],'b--')
            if show_examples:
                plt.plot(1+zvalues_full,(DH_realizations[iperm]/DH0_full).T,'r',alpha=0.5)
            plt.xlabel(r'$1+z$')
            plt.ylabel(r'$D_H(z)/D_H^0(z)$')

            plt.subplot(num_plot_rows,2,2*irow+2)
            plt.xscale('log')
            plt.grid(True)
            plt.xlim([1.,1+np.max(zvalues)])
            plt.fill_between(1+zvalues[1:],DA_ratio_limits[0],DA_ratio_limits[-1],
                facecolor='blue',alpha=0.25)
            plt.plot(1+zvalues[1:],DA_ratio_limits[0],'b--')
            plt.plot(1+zvalues[1:],DA_ratio_limits[1],'b-')
            plt.plot(1+zvalues[1:],DA_ratio_limits[2],'b--')
            if show_examples:
                plt.plot(1+zvalues_full[1:],(DA_realizations[iperm,:,1:]/DA0_full[1:]).T,'r',alpha=0.5)
            plt.xlabel(r'$1+z$')
            plt.ylabel(r'$D_A(z)/D_A^0(z)$')

            irow += 1

        # Plot zooms of DH,DA up to zmax.
        if args.zoom:

            plt.subplot(num_plot_rows,2,2*irow+1)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zvalues[:iend],1e-3*DH_limits[0,:iend],1e-3*DH_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zvalues[:iend],1e-3*DH_limits[0,:iend],'b--')
            plt.plot(zvalues[:iend],1e-3*DH_limits[1,:iend],'b-')
            plt.plot(zvalues[:iend],1e-3*DH_limits[2,:iend],'b--')
            if show_examples:
                plt.plot(zvalues_full[:iend_full],
                    1e-3*DH_realizations[iperm,:,:iend_full].T,'r',alpha=0.5)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$D_H(z)$ (Gpc)')

            plt.subplot(num_plot_rows,2,2*irow+2)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zvalues[:iend],1e-3*DA_limits[0,:iend],1e-3*DA_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zvalues[:iend],1e-3*DA_limits[0,:iend],'b--')
            plt.plot(zvalues[:iend],1e-3*DA_limits[1,:iend],'b-')
            plt.plot(zvalues[:iend],1e-3*DA_limits[2,:iend],'b--')
            if show_examples:
                plt.plot(zvalues_full[:iend_full],
                    1e-3*DA_realizations[iperm,:,:iend_full].T,'r',alpha=0.5)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$D_A(z)$ (Gpc)')

            irow += 1

        # Plot dark-energy diagnostics up to zmax.
        if args.dark_energy:

            # Calculate the corresponding limits and realizations acceleration H(z)/(1+z).
            accel_limits = clight/DH_limits/(1+zvalues)
            if show_examples:
                accel_realizations = clight/DH_realizations[iperm]/(1+zvalues_full)

            plt.subplot(num_plot_rows,2,2*irow+1)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zvalues[:iend],accel_limits[0,:iend],accel_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zvalues[:iend],accel_limits[0,:iend],'b--')
            plt.plot(zvalues[:iend],accel_limits[1,:iend],'b-')
            plt.plot(zvalues[:iend],accel_limits[2,:iend],'b--')
            if show_examples:
                plt.plot(zvalues_full[:iend_full],accel_realizations[:,:iend_full].T,'r',alpha=0.5)
            plt.xlabel(r'$z$')
            plt.ylabel(r'$H(z)/(1+z)$ (Mpc)')

            irow += 1

        if num_plot_rows > 0:
            plt.savefig(args.output + name + '.' + args.plot_format)
            if args.show:
                plt.show()
            plt.close()

if __name__ == '__main__':
    main()
