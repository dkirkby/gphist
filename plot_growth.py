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
        #if num_plot_rows == 0 and not args.nlp and not args.growth:
        #print 'No plots selected.'
        #return 0
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
    from matplotlib import gridspec

    # Load the input file.
    loaded = np.load(args.input + '.npz')
    #f_loaded = np.load('npz_files/phi_test/test2.0.npz')
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


    # Initialize the posterior permutations.
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
    # Initialize the hyperparameter grid.

    Dg_vals_r = np.array([0.67, 0.47, 0.53, 0.37, 0.59])
    Dg_err_r = np.array([0.3, 0.2, 0.2, 0.16, 0.12])
    Dg_z_r = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    Dg_vals_b = np.array([0.72, 0.7, 0.47, 0.2, 0.52])
    Dg_err_b = np.array([0.25, 0.15, 0.15, 0.15, 0.12])
    Dg_z_b = Dg_z_r + 0.02
    print 'making growth / DE combo plot'
    name = 'idk'
    fig = plt.figure(name,figsize=(12,8))
    fig.subplots_adjust(left=0.06,bottom=0.07,right=0.98,
                        top=0.99,wspace=0.0,hspace=0.18)
    fig.set_facecolor('white')
            # Find first z index beyond zmax.

    omega_matter0 = gphist.cosmology.get_omega_matter_evolution(zvalues,DH0)
    omega_matter0 = omega_matter0[::-1]
    plt.plot(zvalues,f0[0,::-1],'k-',label = r'$\Lambda CDM$')
    plt.plot(zvalues,omega_matter0[::-1]**0.55,'k--')

    iend = 1+np.argmax(zvalues > args.zmax)
    plot_colors = ('green','blue','magenta')
    plot_c_alt = ('g', 'b','r')
    plot_c = ('g','b','m')
    alpha_c = (0.25,0.5)
    gs = gridspec.GridSpec(1,2,width_ratios = [2,1])
    for i,j in zip((31,63),(0,1)):
        print posterior_names[perms[i]]
        name = '-'.join(posterior_names[perms[i]])
        DH_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                DH_hist[i],[args.level],bin_range)
        DH_limits = DH_ratio_limits*DH0
        f_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                f_hist[i],[args.level],bin_range)
        f_limits = f_ratio_limits*f0
            
        omega_matter = gphist.cosmology.get_omega_matter_evolution(zvalues,DH_limits[1])
        omega_matter = omega_matter[::-1]
        
            
        def func(gamma):
            return np.sum((f_limits[1,:] - omega_matter**gamma)**2)
            
        res = minimize(func,0.55,method = 'Nelder-Mead')
        gamma_best_fit = res.x
            
      

        plt.subplot(gs[1])
        #plt.subplot(2,2,2)
        plt.xscale('log')
        plt.grid(True)
        plt.xlim([args.zmax,33.])
        print zvalues[iend-1]
        plt.ylim([0.4,1.1])
        #plt.fill_between(lna,f_limits[0],f_limits[-1],facecolor=plot_colors[j],alpha=0.25)
        #plt.plot(lna,f_limits[0,:],plot_c[j]+':')
        #plt.plot(lna,f_limits[1,:],plot_c[j]+'-',label = name)
        #plt.plot(lna,f_limits[2,:],plot_c[j]+':')
        plt.fill_between(zvalues,f_limits[0,::-1],f_limits[-1,::-1],facecolor=plot_colors[j],alpha=alpha_c[j])
        plt.plot(zvalues,f_limits[0,::-1],plot_c[j]+':')
        plt.plot(zvalues,f_limits[1,::-1],plot_c[j]+'-')
        plt.plot(zvalues,f_limits[2,::-1],plot_c[j]+':')        
        
        #plt.plot(lna,omega_matter**gamma_best_fit,plot_c[j]+'--')
        plt.plot(zvalues,omega_matter[::-1]**gamma_best_fit,plot_c[j]+'--')
            #plt.plot(lna,phi0*phi_realizations[iperm,0], label ='Realization 1')
            #plt.plot(lna,phi0*phi_realizations[iperm,1], label ='Realization 2')
        #plt.xlim(-3.5,0)
        #plt.xlim(0,50.)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$1 + d \log (\phi) / d \log(a)$')
        frame1 = plt.gca()
        frame1.axes.yaxis.set_ticklabels([])
        
        
        
        
        plt.subplot(gs[0])
        plt.xscale('linear')
        plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.4,1.1])
        plt.errorbar(Dg_z_r, Dg_vals_r, yerr=Dg_err_r, fmt='ro')
        plt.errorbar(Dg_z_b, Dg_vals_b, yerr=Dg_err_b, fmt='bs')
        plt.fill_between(zvalues,f_limits[0,::-1],f_limits[-1,::-1],facecolor=plot_colors[j],alpha=alpha_c[j])
        plt.plot(zvalues,f_limits[0,::-1],plot_c[j]+':')
        plt.plot(zvalues,f_limits[1,::-1],plot_c[j]+'-')
        plt.plot(zvalues,f_limits[2,::-1],plot_c[j]+':')        
        
        #plt.plot(lna,omega_matter**gamma_best_fit,plot_c[j]+'--')
        plt.plot(zvalues,omega_matter[::-1]**gamma_best_fit,plot_c[j]+'--')
        plt.xlabel(r'$z$')
        plt.ylabel(r'$1 + d \log (\phi) / d \log(a)$')
        
    green_patch = mpatches.Patch(color=plot_colors[0],alpha=alpha_c[0],label = 'GP Result')
    blue_patch = mpatches.Patch(color=plot_colors[1],alpha=alpha_c[1],label = 'GP Result + DESI')
    #red_patch = mpatches.Patch(color = plot_colors[2],alpha=alpha_c[2],label = 'CMB-SN-LRG')
    plt.legend(handles = [green_patch,blue_patch],loc='best')
    #plt.legend(loc='best')
    plt.savefig(args.output + '_growth_combo' + '.' + args.plot_format)
    plt.close()











if __name__ == '__main__':
    main()
