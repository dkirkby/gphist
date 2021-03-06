#!/usr/bin/env python
"""Plot expansion history inferences.
"""

import argparse

import numpy as np
# matplotlib is imported inside main()
from scipy.optimize import minimize

import gphist

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',type = str, default = None,
        help = 'name of input file to read (extension .npz will be added)')
    parser.add_argument('--posterior', type = str, action='append', metavar = 'NAME',
        help = 'posteriors to plot (can be repeated, plot all if omitted)')
    parser.add_argument('--nlp', action = 'store_true',
        help = 'show plots of posterior -log(P) marginalized over hyperparameters')
    parser.add_argument('--full', action = 'store_true',
        help = 'show plots of DH/DH0,DA/DA0 evolution over the full redshift range')
    parser.add_argument('--zoom', action = 'store_true',
        help = 'show plots of DH,DA on a linear scale up to redshift zmax')
    parser.add_argument('--dark-energy', action = 'store_true',
        help = 'show plots of H(z)/(1+z) and Omega_DE(z)/Omega_DE(0) up to redshift zmax')
    parser.add_argument('--growth', action = 'store_true',
        help = 'show plots of phi(lna)')        
    parser.add_argument('--zmax', type = float, default = 3.0,
        help = 'maximum redshift to display on H(z)/(1+z) plot')
    parser.add_argument('--level', type = float, default = 0.9,
        help = 'confidence level to plot')
    parser.add_argument('--examples', action = 'store_true',
        help = 'include examples of random realizations in each plot')
    parser.add_argument('--output', type = str, default = None,
        help = 'base name for saving plots (no plots are saved if not set)')
    parser.add_argument('--show', action = 'store_true',
        help = 'show each plot (in addition to saving it if output is set)')
    parser.add_argument('--plot-format', type = str, default = 'png', metavar = 'FMT',
        help = 'format for saving plots (png,pdf,...)')
    args = parser.parse_args()

    # Do we have any inputs to read?
    if args.input is None:
        print 'Missing required input arg.'
        return -1

    # Do we have anything to plot?
    num_plot_rows = args.full + args.zoom + args.dark_energy
    if num_plot_rows == 0 and not args.nlp and not args.growth:
        print 'No plots selected.'
        return 0
    if not args.output and not args.show:
        print 'No output requested.'
    if args.examples:
        print 'Option --examples not implemented yet.'
        return -1

    # Initialize matplotlib.
    import matplotlib as mpl
    if not args.show:
        # Use the default backend, which does not require X11 on unix systems.
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Load the input file.
    loaded = np.load(args.input + '.npz')
    DH_hist = loaded['DH_hist']
    DA_hist = loaded['DA_hist']
    de_hist = loaded['de_hist']
    f_hist = loaded['phi_hist']
    #print 'the shape of the combined phi histogram is '+str(phi_hist.shape)
    #print 'the shape of the combined DH histogram is '+str(DH_hist.shape)
    #phi_realizations = loaded['phi_realizations']
    DH0 = loaded['DH0']
    DA0 = loaded['DA0']
    de0 = loaded['de0']
    f0 = loaded['phi0']
    zvalues = loaded['zvalues']
    lna=-np.log(1 + zvalues[::-1])
    fixed_options = loaded['fixed_options']
    bin_range = loaded['bin_range']
    hyper_range = loaded['hyper_range']
    posterior_names = loaded['posterior_names']
    # The -log(P) array is only present if this file was written by combine.py
    hyper_nlp = None
    if args.nlp:
        if 'hyper_nlp' in loaded.files:
            hyper_nlp = loaded['hyper_nlp']
        else:
            print 'Ignoring option --nlp since -log(P) values are not available.'
            args.nlp = False
    # Check that dark energy evolution is available if plots are requested.
    if args.dark_energy and de_hist is None:
        print 'Input file is missing dark-energy evolution'
        return -1

    # Initialize the posterior permutations.
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
    # Initialize the hyperparameter grid.
    n_samples,n_h,n_sigma = fixed_options
    h_min,h_max,sigma_min,sigma_max = hyper_range
    hyper_grid = gphist.process.HyperParameterLogGrid(
        n_h,h_min,h_max,n_sigma,sigma_min,sigma_max)

    # Initialize -log(P) plotting.
    if args.nlp:
        # Factor of 0.5 since -logP = 0.5*chisq
        nlp_levels = 0.5*gphist.analysis.get_delta_chisq(num_dof=2)
        print nlp_levels

    # Loop over posterior permutations.
    for iperm,perm in enumerate(perms):

        name = '-'.join(posterior_names[perms[iperm]]) or 'Prior'
        if args.posterior and name not in args.posterior:
            continue
        print '%d : %s' % (iperm,name)

        if args.nlp:
            fig = plt.figure('NLP-'+name,figsize=(9,7))
            fig.subplots_adjust(left=0.10,bottom=0.07,right=1.00,top=0.98)
            fig.set_facecolor('white')
            plt.xscale('log')
            plt.yscale('log')
            if iperm > 0:
                missing = hyper_nlp[iperm] == 0
                nlp_min = np.min(hyper_nlp[iperm,np.logical_not(missing)])
                nlp = np.ma.array(hyper_nlp[iperm]-nlp_min,mask=missing)
            else:
                # The prior distribution should be flat by construction.
                nlp = hyper_nlp[iperm]
            plt.pcolormesh(hyper_grid.sigma_edges,hyper_grid.h_edges,nlp,
                cmap='rainbow',vmin=0.,vmax=50.,rasterized=True)
            if iperm > 0: # Cannot plot contours for the flat prior.
                plt.colorbar()
                plt.contour(hyper_grid.sigma,hyper_grid.h,nlp,colors='w',
                    levels=nlp_levels,linestyles=('-','--',':'))
            plt.xlabel(r'Hyperparameter $\sigma$')
            plt.ylabel(r'Hyperparameter $h$')
            if args.output:
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

            fig = plt.figure(name,figsize=(12,4*num_plot_rows))
            fig.subplots_adjust(left=0.06,bottom=0.07,right=0.98,
                top=0.99,wspace=0.18,hspace=0.18)
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
            plt.plot(1+zvalues,DH_ratio_limits[0],'b:')
            plt.plot(1+zvalues,DH_ratio_limits[1],'b-')
            plt.plot(1+zvalues,DH_ratio_limits[2],'b:')
            plt.xlabel(r'$1+z$')
            plt.ylabel(r'$D_H(z)/D_H^0(z)$')

            plt.subplot(num_plot_rows,2,2*irow+2)
            plt.xscale('log')
            plt.grid(True)
            plt.xlim([1.,1+np.max(zvalues)])
            plt.fill_between(1+zvalues[1:],DA_ratio_limits[0],DA_ratio_limits[-1],
                facecolor='blue',alpha=0.25)
            plt.plot(1+zvalues[1:],DA_ratio_limits[0],'b:')
            plt.plot(1+zvalues[1:],DA_ratio_limits[1],'b-')
            plt.plot(1+zvalues[1:],DA_ratio_limits[2],'b:')
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
            plt.plot(zvalues[:iend],1e-3*DH_limits[0,:iend],'b:')
            plt.plot(zvalues[:iend],1e-3*DH_limits[1,:iend],'b-')
            plt.plot(zvalues[:iend],1e-3*DH_limits[2,:iend],'b:')
            plt.plot(zvalues[:iend],1e-3*DH0[:iend],'g--')
            plt.xlabel(r'$z$')
            plt.ylabel(r'$D_H(z)$ (Gpc)')

            plt.subplot(num_plot_rows,2,2*irow+2)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zvalues[:iend],1e-3*DA_limits[0,:iend],1e-3*DA_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zvalues[:iend],1e-3*DA_limits[0,:iend],'b:')
            plt.plot(zvalues[:iend],1e-3*DA_limits[1,:iend],'b-')
            plt.plot(zvalues[:iend],1e-3*DA_limits[2,:iend],'b:')
            plt.plot(zvalues[:iend],1e-3*DA0[:iend],'g--')
            plt.xlabel(r'$z$')
            plt.ylabel(r'$D_A(z)$ (Gpc)')

            irow += 1
   


        # Plot dark-energy evolution up to zmax.
        if args.dark_energy:

            '''
            # Calculate the acceleration H(z)/(1+z).
            accel_limits = gphist.distance.get_acceleration(zvalues,DH_limits)

            plt.subplot(num_plot_rows,2,2*irow+1)
            plt.xscale('linear')
            plt.grid(True)
            plt.xlim([0.,args.zmax])
            plt.fill_between(zvalues[:iend],accel_limits[0,:iend],accel_limits[-1,:iend],
                facecolor='blue',alpha=0.25)
            plt.plot(zvalues[:iend],accel_limits[0,:iend],'b:')
            plt.plot(zvalues[:iend],accel_limits[1,:iend],'b-')
            plt.plot(zvalues[:iend],accel_limits[2,:iend],'b:')
            plt.xlabel(r'$z$')
            plt.ylabel(r'$H(z)/(1+z)$ (Mpc)')
            '''

            de_labels = (r'$\omega_{\phi}(z)/h_0^2$',r'$\omega_{\phi}(z)/h(z)^2$')
            assert len(de_labels) == len(de_hist), 'Mismatch in number of dark-energy variables'

            # Bin range for DE histograms is hardcoded for now.
            de_bin_range = np.array([-10.,+10.])

            for ide in range(len(de_hist)):

                de_limits = gphist.analysis.calculate_confidence_limits(
                    de_hist[ide,iperm,:iend],[args.level],de_bin_range)

                plt.subplot(num_plot_rows,2,2*irow+ide+1)
                plt.xscale('linear')
                plt.grid(True)
                plt.xlim([0.,args.zmax])
                #plt.ylim([0.,None])
                plt.fill_between(zvalues[:iend],de_limits[0],de_limits[-1],
                    facecolor='blue',alpha=0.25)
                plt.plot(zvalues[:iend],de_limits[0],'b:')
                plt.plot(zvalues[:iend],de_limits[1],'b-')
                plt.plot(zvalues[:iend],de_limits[2],'b:')
                plt.plot(zvalues[:iend],de0[ide,:iend],'g--')
                plt.xlabel(r'$z$')
                plt.ylabel(de_labels[ide])

            irow += 1

        if num_plot_rows > 0:
            if args.output:
                plt.savefig(args.output + name + '.' + args.plot_format)
            if args.show:
                plt.show()
            plt.close()
    
    # Plot growth function phi.
        if args.growth:
            #print phi_hist.shape
            #plt.plot(range(phi_hist.shape[2]), phi_hist[iperm,50,:])
            #plt.savefig('zzz_hist_'+str(iperm)+'.png')
            #plt.close()
            #stuff here needs DH_limits which are calculated only if the --full arguments are passed
            #print phi_realizations.shape
            f_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                             f_hist[iperm],[args.level],bin_range)
            f_limits = f_ratio_limits*f0
        
            omega_matter = gphist.cosmology.get_omega_matter_evolution(zvalues,DH_limits[1])
            omega_matter = omega_matter[::-1]
            omega_matter0 = gphist.cosmology.get_omega_matter_evolution(zvalues,DH0)
            omega_matter0 = omega_matter0[::-1]

            def func(gamma):
                return np.sum((f_limits[1,:] - omega_matter**gamma)**2)

            res = minimize(func,0.55,method = 'Nelder-Mead')
            gamma_best_fit = res.x

            plt.grid(True)
            plt.fill_between(lna,f_limits[0],f_limits[-1],facecolor='blue',alpha=0.25)
            plt.plot(lna,f_limits[0,:],'b:')
            plt.plot(lna,f_limits[1,:],'b-',label = 'Best Fit')
            plt.plot(lna,f_limits[2,:],'b:')
            plt.plot(lna,f0[0],'k',label = r'$\Lambda CDM$')
            plt.plot(lna,omega_matter**gamma_best_fit,'r',label = r'$\Omega_m^\gamma , \gamma = $ ' +str(gamma_best_fit))
            plt.plot(lna,omega_matter0**0.55,'c',label=r'$\Omega_{m, \Lambda CDM}^{0.55}$')
            #plt.plot(lna,phi0*phi_realizations[iperm,0], label ='Realization 1')
            #plt.plot(lna,phi0*phi_realizations[iperm,1], label ='Realization 2')
            plt.xlim(-3.5,0)
            plt.xlabel(r'$\log (a)$')
            plt.ylabel(r'$1 + d \log (\phi) / d \log(a)$')
            plt.legend(loc='best')
            plt.savefig(args.output + '_growth-' + name + '.' + args.plot_format)
            plt.close()





if __name__ == '__main__':
    main()
