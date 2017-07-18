#!/usr/bin/env python
"""Plot expansion history inferences.
"""

import argparse

import numpy as np
# matplotlib is imported inside main()
from scipy.optimize import minimize
from scipy.interpolate import interp1d

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
    from matplotlib import rc
    if not args.show:
        # Use the default backend, which does not require X11 on unix systems.
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import gridspec

    # Load the input file.
    loaded = np.load(args.input + '.npz')
    DH_hist = loaded['DH_hist']
    DA_hist = loaded['DA_hist']
    de_hist = loaded['de_hist']
    f_hist = loaded['phi_hist']
   
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


    print 'making combo plot'
    sigma_8 = 0.830
    print r'using $\sigma_8 = 0.83$'

    Dg_vals_r = np.array([0.67, 0.47, 0.53, 0.37, 0.59])
    Dg_err_r = np.array([0.3, 0.2, 0.2, 0.16, 0.12])
    Dg_z_r = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    Dg_vals_b = np.array([0.72, 0.7, 0.47, 0.2, 0.52])
    Dg_err_b = np.array([0.25, 0.15, 0.15, 0.15, 0.12])
    Dg_z_b = Dg_z_r + 0.02
    omega_matter0 = gphist.cosmology.get_omega_matter_evolution(zvalues,DH0)
    omega_matter0 = omega_matter0[::-1]

    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.size': 18})
    name = 'idk'
    fig = plt.figure(name,figsize=(12,8))
    fig.subplots_adjust(left=0.075,bottom=0.05,right=0.98,
                        top=0.975,wspace=0.0,hspace=0.18)
    fig.set_facecolor('white')
    # Find first z index beyond zmax.
    iend = 1+np.argmax(zvalues > args.zmax)
    plot_colors = ('blue','green','red')
    plot_c = ('b','g','r')
    alpha_c = (0.15,0.25,0.15)
    gs = gridspec.GridSpec(1,1)
    #gs1 = gridspec.GridSpec(3,2,width_ratios = [2,1])
    #gs.update(hspace=0.)
    #gs1.update(hspace=0.3)
    for i,j in zip((31,3,28),(0,1,2)):
        print posterior_names[perms[i]]
        name = '-'.join(posterior_names[perms[i]])
        DH_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                    DH_hist[i],[args.level],bin_range)
        DA_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                    DA_hist[i],[args.level],bin_range)
        f_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                f_hist[i],[args.level],bin_range)
        
        interp_DH_low = interp1d(zvalues[:iend],DH_ratio_limits[0,:iend],kind='cubic')
        interp_DH_mid = interp1d(zvalues[:iend],DH_ratio_limits[1,:iend],kind='cubic')
        interp_DH_high = interp1d(zvalues[:iend],DH_ratio_limits[2,:iend],kind='cubic')
        f_limits = f_ratio_limits*f0
        f_limits_low_ordered = f_limits[0,::-1]
        f_limits_mid_ordered = f_limits[1,::-1]
        f_limits_high_ordered = f_limits[2,::-1]
        filter_f_low = interp1d(zvalues[::3],f_limits_low_ordered[::3],kind='cubic')
        filter_f_mid = interp1d(zvalues[::3],f_limits_mid_ordered[::3],kind='cubic')
        filter_f_high = interp1d(zvalues[::3],f_limits_high_ordered[::3],kind='cubic')
        DH_limits = DH_ratio_limits*DH0
        DA_limits = np.empty_like(DH_limits)
        DA_limits[:,1:] = DA_ratio_limits*DA0[1:]
        DA_limits[:,0] = 0.
        
        omega_matter = gphist.cosmology.get_omega_matter_evolution(zvalues,DH_limits[1])
        omega_matter = omega_matter[::-1]
        
        def func(gamma):
            return np.sum((f_limits[1,:] - omega_matter**gamma)**2)
            
        res = minimize(func,0.55,method = 'Nelder-Mead')
        gamma_best_fit = res.x








#growth
        plt.subplot(gs[0])
        ax4 = plt.subplot(gs[0])
        ax4.set_yticks([0.0,0.2,0.4, 0.6, 0.8, 1.0,])
        #gs.update(hspace=0.18)
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.0,1.0])
        plt.plot(zvalues,sigma_8*f0[0,::-1],'k-',label = r'$\Lambda CDM$')
        plt.plot(zvalues,sigma_8*omega_matter0[::-1]**0.55,'k--')
        #plt.fill_between(zvalues,f_limits[0,::-1],f_limits[-1,::-1],
        #                facecolor=plot_colors[j],alpha=alpha_c[j])
        plt.fill_between(zvalues[:iend],sigma_8*filter_f_low(zvalues[:iend]),sigma_8*filter_f_high(zvalues[:iend]),
                        facecolor=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:')
        plt.errorbar(Dg_z_r, Dg_vals_r, yerr=Dg_err_r, fmt='ro')
        plt.errorbar(Dg_z_b, Dg_vals_b, yerr=Dg_err_b, fmt='bs')
        plt.plot(zvalues[:iend],sigma_8*filter_f_low(zvalues[:iend]),plot_c[j]+'-')
        plt.plot(zvalues[:iend],sigma_8*filter_f_mid(zvalues[:iend]),plot_c[j]+'-')
        plt.plot(zvalues[:iend],sigma_8*filter_f_high(zvalues[:iend]),plot_c[j]+'-')
        #plt.plot(zvalues,f_limits[0,::-1],plot_c[j]+'-')
        #plt.plot(zvalues,f_limits[1,::-1],plot_c[j]+'-')
        #plt.plot(zvalues,f_limits[2,::-1],plot_c[j]+'-')
        plt.plot(zvalues,sigma_8*omega_matter[::-1]**gamma_best_fit,plot_c[j]+'--')
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$\sigma_8 (1 + d \log (\phi) / d \log(a))$',fontsize=20)

    blue_patch = mpatches.Patch(color=plot_colors[0],alpha=alpha_c[0],label = r'H0-Ly$\alpha$-CMB-SN-LRG')
    green_patch = mpatches.Patch(color=plot_colors[2],alpha=alpha_c[2],label = r'H0-Ly$\alpha$')
    red_patch = mpatches.Patch(color = plot_colors[1],alpha=alpha_c[1],label = 'CMB-SN-LRG')
    plt.subplot(gs[0])
    plt.legend(handles = [green_patch,red_patch,blue_patch],loc='upper left',prop={'size':18})

    plt.savefig(args.output + '.' + args.plot_format)
    plt.close()











if __name__ == '__main__':
    main()
