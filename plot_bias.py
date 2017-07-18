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
    if not args.show:
        # Use the default backend, which does not require X11 on unix systems.
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import gridspec
    from matplotlib import rc

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


    print 'making combo plot'
    omega_matter0 = gphist.cosmology.get_omega_matter_evolution(zvalues,DH0)
    omega_matter0 = omega_matter0[::-1]


    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.size': 18})
    name = 'idk'
    input_data = np.loadtxt('bias_data.txt')
    z_values_input = input_data[0]
    DCw_input = input_data[1]
    DC0_input = input_data[2]
    DHw_input = input_data[3]
    DH0_input = input_data[4]
    fw_input = input_data[5]
    f0_input = input_data[6]
    fig = plt.figure(name,figsize=(12,12))
    fig.subplots_adjust(left=0.085,bottom=0.06,right=0.98,
                        top=0.975,wspace=0.0,hspace=0.18)
    fig.set_facecolor('white')
    #matplotlib.rcParams.update({'font.size': 22})

    # Find first z index beyond zmax.
    iend = 1+np.argmax(zvalues > args.zmax)
    #plot_colors = ('blue','green','red')
    plot_colors = [ np.array((55,126,184))/255., np.array((152,78,163))/255., np.array((255,128,0))/255.]
    #plot_c = ('b','g','r')
    alpha_c = (0.6,0.6,0.6)
    gs = gridspec.GridSpec(3,2,width_ratios = [2,1])
    for i,j in zip((23,),(2,)):
        print posterior_names[perms[i]]
        name = '-'.join(posterior_names[perms[i]])
        DH_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                DH_hist[i],[args.level],bin_range)
        DA_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                DA_hist[i],[args.level],bin_range)
        f_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                f_hist[i],[args.level],bin_range)
        f_limits = f_ratio_limits*f0
        interp_DH_low = interp1d(zvalues[:iend+1],DH_ratio_limits[0,:iend+1],kind='cubic')
        interp_DH_mid = interp1d(zvalues[:iend+1],DH_ratio_limits[1,:iend+1],kind='cubic')
        interp_DH_high = interp1d(zvalues[:iend+1],DH_ratio_limits[2,:iend+1],kind='cubic')
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
        print gamma_best_fit

#plotting the log parts of the split lin-log plot
# DH
        plt.subplot(gs[1])
        plt.xscale('log')
        #plt.grid(True)
        plt.xlim([args.zmax,np.max(zvalues)])
        plt.ylim([0.85,1.26])
        plt.fill_between(zvalues[4::5],DH_ratio_limits[0,4::5],DH_ratio_limits[-1,4::5],
                color=plot_colors[j],alpha=alpha_c[j])
        
        #plt.plot(zvalues[1:],spline_DH_low(zvalues[1:]),plot_c[j]+'-')
        #plt.plot(zvalues[1:],spline_DH_mid(zvalues[1:]),plot_c[j]+'-')
        #plt.plot(zvalues[1:],spline_DH_high(zvalues[1:]),plot_c[j]+'-')
        #plt.plot(zvalues[4::5],DH_ratio_limits[0,4::5],plot_c[j]+'-')
        plt.plot(zvalues[4::5],DH_ratio_limits[1,4::5],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues[4::5],DH_ratio_limits[2,4::5],plot_c[j]+'-')
        plt.plot(z_values_input[1:],DHw_input[1:]/DH0_input[1:],'k',linewidth=3.0)
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        plt.xlabel(r'$z$',fontsize=20)
        frame1 = plt.gca()
        frame1.axes.yaxis.set_ticklabels([])

#DA
        plt.subplot(gs[3])
        ax3 = plt.subplot(gs[3])
        ax3.set_yticks([1.00, 1.04, 1.08, 1.12])
        plt.xscale('log')
        #plt.grid(True)
        plt.xlim([args.zmax,np.max(zvalues)])
        plt.ylim([0.995,1.145])
        plt.fill_between(zvalues[1:],DA_ratio_limits[0],DA_ratio_limits[-1],
                    color=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues[1::2],DA_ratio_limits[0,::2],plot_c[j]+'-')
        plt.plot(zvalues[1::2],DA_ratio_limits[1,::2],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues[1::2],DA_ratio_limits[2,::2],plot_c[j]+'-')
        plt.plot(z_values_input[1:], DCw_input[1:]/DC0_input[1:],'k',linewidth=3.0)
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        plt.xlabel(r'$z$',fontsize=20)
        frame1 = plt.gca()
        frame1.axes.yaxis.set_ticklabels([])
        
#growth
        plt.subplot(gs[5])
        ax5 = plt.subplot(gs[5])
        ax5.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
        plt.xscale('log')
        #plt.grid(True)
        plt.xlim([args.zmax,33.])
        plt.ylim([0.39,1.22])
        #plt.plot(zvalues,f0[0,::-1],'k-',label = r'$\Lambda CDM$')
        #plt.plot(zvalues,omega_matter0[::-1]**0.55,'k--')
        plt.fill_between(zvalues[::2],f_limits[0,::-2],f_limits[-1,::-2],color=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues[::2],f_limits[0,::-2],plot_c[j]+'-')
        plt.plot(zvalues[::2],f_limits[1,::-2],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues[::2],f_limits[2,::-2],plot_c[j]+'-')
        plt.plot(zvalues[::1],omega_matter[::-1]**gamma_best_fit,color=plot_colors[j],linestyle='--',linewidth=3.0)
        plt.plot(z_values_input,fw_input,'k-',linewidth=3.0)
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$1 + d \log (\phi) / d \log(a)$')
        frame1 = plt.gca()
        frame1.axes.yaxis.set_ticklabels([])
                             

    
# Plotting the linear parts of the split lin-log plot
#DH
        plt.subplot(gs[0])
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.85,1.26])
        plt.fill_between(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_low(np.linspace(0,zvalues[iend],num=100,endpoint=True)),interp_DH_high(np.linspace(0,zvalues[iend],num=100,endpoint=True)),
                         color=plot_colors[j],alpha=alpha_c[j])
        plt.plot(z_values_input[1:],DHw_input[1:]/DH0_input[1:],'k',linewidth=3.0)
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        #plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_low(np.linspace(0,zvalues[iend],num=100,endpoint=True)),plot_c[j]+'-')
        plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_mid(np.linspace(0,zvalues[iend],num=100,endpoint=True)),color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_high(np.linspace(0,zvalues[iend],num=100,endpoint=True)),plot_c[j]+'-')
        #plt.plot(zvalues[:iend],DH_ratio_limits[0,:iend],plot_c[j]+'-')
        #plt.plot(zvalues[:iend],DH_ratio_limits[1,:iend],plot_c[j]+'-')
        #plt.plot(zvalues[:iend],DH_ratio_limits[2,:iend],plot_c[j]+'-')
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$D_H(z)/D_H^0(z)$',fontsize=20)
 
#DA
        plt.subplot(gs[2])
        ax2 = plt.subplot(gs[2])
        ax2.set_yticks([1.00, 1.04, 1.08, 1.12])
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.995,1.145])
        plt.fill_between(zvalues[1:iend],DA_ratio_limits[0,:iend-1],DA_ratio_limits[-1,:iend-1],
                         color=plot_colors[j],alpha=alpha_c[j])
        plt.plot(z_values_input[1:], DCw_input[1:]/DC0_input[1:],'k',linewidth=3.0)
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        #plt.plot(zvalues[1:iend],DA_ratio_limits[0,:iend-1],plot_c[j]+'-')
        plt.plot(zvalues[1:iend],DA_ratio_limits[1,:iend-1],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues[1:iend],DA_ratio_limits[2,:iend-1],plot_c[j]+'-')
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$D_A(z)/D_A^0(z)$',fontsize=20)

#growth
        plt.subplot(gs[4])
        ax4 = plt.subplot(gs[4])
        ax4.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.39,1.22])
        #plt.plot(zvalues,f0[0,::-1],'k-',label = r'$\Lambda CDM$')
        #plt.plot(zvalues,omega_matter0[::-1]**0.55,'k--')
        plt.plot(z_values_input,fw_input,'k-',linewidth=3.0)
        plt.fill_between(zvalues,f_limits[0,::-1],f_limits[-1,::-1],
                        color=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues,f_limits[0,::-1],plot_c[j]+'-')
        plt.plot(zvalues,f_limits[1,::-1],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues,f_limits[2,::-1],plot_c[j]+'-')
        plt.plot(zvalues,omega_matter[::-1]**gamma_best_fit,color=plot_colors[j],linestyle='--',linewidth=3.0)
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$1 + d \log (\phi) / d \log(a)$',fontsize=20)

    blue_patch = mpatches.Patch(color=plot_colors[2],alpha=alpha_c[2],label = r'With mock w$_0$w$_a$ data')
    #green_patch = mpatches.Patch(color='k',label = r'Input $w_0 w_a$')
    #red_patch = mpatches.Patch(color = plot_colors[2],alpha=alpha_c[2],label = 'CMB-SN-LRG')
    plt.subplot(gs[0])
    plt.legend(handles = [blue_patch],loc='upper left',frameon=False)

    plt.savefig(args.output +  '.' + args.plot_format)
    plt.close()




if __name__ == '__main__':
    main()
