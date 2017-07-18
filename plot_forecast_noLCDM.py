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
    loaded = np.load('final.npz')
    DH_hist = loaded['DH_hist']
    DA_hist = loaded['DA_hist']
    de_hist = loaded['de_hist']
    f_hist = loaded['f_hist']

    
    #f_hist = loaded['phi_hist']
    #print 'the shape of the combined phi histogram is '+str(phi_hist.shape)
    #print 'the shape of the combined DH histogram is '+str(DH_hist.shape)
    #phi_realizations = loaded['phi_realizations']
    DH0 = loaded['DH0']
    DA0 = loaded['DA0']
    de0 = loaded['de0']
    f0 = loaded['f0']
    zvalues = loaded['zvalues']
    lna=-np.log(1 + zvalues[::-1])
    fixed_options = loaded['fixed_options']
    bin_range = loaded['bin_range']
  
    posterior_names = loaded['posterior_names']
    # The -log(P) array is only present if this file was written by combine.py


    # Initialize the posterior permutations.
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
    # Initialize the hyperparameter grid.
    n_samples,n_h,n_sigma = fixed_options





    print 'making combo plot'
#59 corresponds to H0-Lya-CMB-SN-LRG
#3 corresponds to H0-Lya
#56 corresponds to CMB-SN-LRG
    omega_matter0 = gphist.cosmology.get_omega_matter_evolution(zvalues,DH0)
    omega_matter0 = omega_matter0[::-1]
    
    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.size': 18})
    name = 'idk'
    fig = plt.figure(name,figsize=(12,12))
    fig.subplots_adjust(left=0.075,bottom=0.06,right=0.98,
                        top=0.975,wspace=0.0,hspace=0.18)
    fig.set_facecolor('white')
    # Find first z index beyond zmax.
    iend = 1+np.argmax(zvalues > args.zmax)
    plot_colors = [ np.array((55,126,184))/255., np.array((152,78,163))/255., np.array((255,153,51))/255.]
    #plot_colors = ('red','green','blue')
    #plot_c = ('r','g','b')
    alpha_c = (0.6,0.6,0.5)
    gs = gridspec.GridSpec(3,2,width_ratios = [2,1])
    # plots for the GP-result case
    for i,j in zip((31,63),(0,1)):
        print posterior_names[perms[i]]
        name = '-'.join(posterior_names[perms[i]])
        DH_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                    DH_hist[i],[args.level],bin_range)
        DA_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                    DA_hist[i],[args.level],bin_range)
        f_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                                    f_hist[i],[args.level],bin_range)
        f_limits = f_ratio_limits*f0
        f_limits_low_ordered = f_limits[0,::-1]
        f_limits_mid_ordered = f_limits[1,::-1]
        f_limits_high_ordered = f_limits[2,::-1]
        interp_DH_low = interp1d(zvalues[:iend+1],DH_ratio_limits[0,:iend+1],kind='cubic')
        interp_DH_mid = interp1d(zvalues[:iend+1],DH_ratio_limits[1,:iend+1],kind='cubic')
        interp_DH_high = interp1d(zvalues[:iend+1],DH_ratio_limits[2,:iend+1],kind='cubic')
        interp_f_low = interp1d(zvalues[:iend+1],f_limits_low_ordered[:iend+1],kind='cubic')
        interp_f_mid = interp1d(zvalues[:iend+1],f_limits_mid_ordered[:iend+1],kind='cubic')
        interp_f_high = interp1d(zvalues[:iend+1],f_limits_high_ordered[:iend+1],kind='cubic')
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
        #print zvalues[iend-1]
        plt.ylim([0.942,1.059])
        plt.fill_between(zvalues[2::3],DH_ratio_limits[0,2::3],DH_ratio_limits[-1,2::3],
                color=plot_colors[j],alpha=alpha_c[j])
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        #plt.plot(zvalues[2::3],DH_ratio_limits[0,2::3],plot_c[j]+'-')
        plt.plot(zvalues[2::3],DH_ratio_limits[1,2::3],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues[2::3],DH_ratio_limits[2,2::3],plot_c[j]+'-')
        plt.xlabel(r'$z$',fontsize=20)
        frame1 = plt.gca()
        frame1.axes.yaxis.set_ticklabels([])
        
# DA
        plt.subplot(gs[3])
        plt.xscale('log')
        #plt.grid(True)
        plt.xlim([args.zmax,np.max(zvalues)])
        plt.ylim([0.945,1.007])
        plt.fill_between(zvalues[1:],DA_ratio_limits[0],DA_ratio_limits[-1],
                    color=plot_colors[j],alpha=alpha_c[j])
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        #plt.plot(zvalues[1:],DA_ratio_limits[0],plot_c[j]+'-')
        plt.plot(zvalues[1:],DA_ratio_limits[1],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues[1:],DA_ratio_limits[2],plot_c[j]+'-')
        plt.xlabel(r'$z$',fontsize=20)
        frame1 = plt.gca()
        frame1.axes.yaxis.set_ticklabels([])

# growth
        plt.subplot(gs[5])
        plt.xscale('log')
        #plt.grid(True)
        plt.xlim([args.zmax,33.])
        plt.ylim([0.45,1.08])
        plt.plot(zvalues,f0[0,::-1],'k-',linewidth=3.0)
        plt.plot(zvalues,omega_matter0[::-1]**0.55,'k--',linewidth=3.0)
        plt.fill_between(zvalues,f_limits[0,::-1],f_limits[-1,::-1],color=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues,f_limits[0,::-1],plot_c[j]+'-')
        plt.plot(zvalues,f_limits[1,::-1],color=plot_colors[j],linestyle='-')
        #plt.plot(zvalues,f_limits[2,::-1],plot_c[j]+'-')
        plt.plot(zvalues,omega_matter[::-1]**gamma_best_fit,color=plot_colors[j],linestyle='--',linewidth=3.0)
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$1 + d \log (\phi) / d \log(a)$')
        frame1 = plt.gca()
        frame1.axes.yaxis.set_ticklabels([])
    

# Plotting the linear parts of the split lin-log plot
# DH
        plt.subplot(gs[0])
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.942,1.059])
        #plt.fill_between(zvalues[:iend],DH_ratio_limits[0,:iend],DH_ratio_limits[-1,:iend],
        #                 facecolor=plot_colors[j],alpha=alpha_c[j])
        plt.fill_between(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_low(np.linspace(0,zvalues[iend],num=100,endpoint=True)),interp_DH_high(np.linspace(0,zvalues[iend],num=100,endpoint=True)),
                         color=plot_colors[j],alpha=alpha_c[j])
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        #plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_low(np.linspace(0,zvalues[iend],num=100,endpoint=True)),plot_c[j]+'-')
        plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_mid(np.linspace(0,zvalues[iend],num=100,endpoint=True)),color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_DH_high(np.linspace(0,zvalues[iend],num=100,endpoint=True)),plot_c[j]+'-')
        #plt.plot(zvalues[:iend],DH_ratio_limits[0,:iend],plot_c[j]+'-')
        #plt.plot(zvalues[:iend],DH_ratio_limits[1,:iend],plot_c[j]+'-')
        #plt.plot(zvalues[:iend],DH_ratio_limits[2,:iend],plot_c[j]+'-')
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$D_H(z)/D_H^0(z)$',fontsize=20)

# DA
        plt.subplot(gs[2])
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.945,1.007])
        plt.fill_between(zvalues[1:iend],DA_ratio_limits[0,:iend-1],DA_ratio_limits[-1,:iend-1],
                         color=plot_colors[j],alpha=alpha_c[j])
        plt.plot(zvalues[1:],np.ones(len(zvalues[1:])),'k:',linewidth=3.0)
        #plt.plot(zvalues[1:iend],DA_ratio_limits[0,:iend-1],plot_colors[j]+'-')
        plt.plot(zvalues[1:iend],DA_ratio_limits[1,:iend-1],color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(zvalues[1:iend],DA_ratio_limits[2,:iend-1],plot_colors[j]+'-')
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$D_A(z)/D_A^0(z)$',fontsize=20)

# growth
        plt.subplot(gs[4])
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        plt.ylim([0.45,1.08])
        plt.plot(zvalues,f0[0,::-1],'k-',linewidth=3.0)
        plt.plot(zvalues,omega_matter0[::-1]**0.55,'k--',linewidth=3.0)
        #plt.fill_between(zvalues,f_limits[0,::-1],f_limits[-1,::-1],
        #                facecolor=plot_colors[j],alpha=alpha_c[j])
        plt.fill_between(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_f_low(np.linspace(0,zvalues[iend],num=100,endpoint=True)),interp_f_high(np.linspace(0,zvalues[iend],num=100,endpoint=True)),color=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues,f_limits[0,::-1],plot_c[j]+'-')
        #plt.plot(zvalues,f_limits[1,::-1],plot_c[j]+'-')
        #plt.plot(zvalues,f_limits[2,::-1],plot_c[j]+'-')
        #plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_f_low(np.linspace(0,zvalues[iend],num=100,endpoint=True)),plot_c[j]+'-')
        plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_f_mid(np.linspace(0,zvalues[iend],num=100,endpoint=True)),color=plot_colors[j],linestyle='-',linewidth=3.0)
        #plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_f_high(np.linspace(0,zvalues[iend],num=100,endpoint=True)),plot_c[j]+'-')
        plt.plot(zvalues,omega_matter[::-1]**gamma_best_fit,color=plot_colors[j],linestyle='--',linewidth=3.0)
        plt.xlabel(r'$z$',fontsize=20)
        plt.ylabel(r'$1 + d \log (\phi) / d \log(a)$',fontsize=20)








    green_patch = mpatches.Patch(color=plot_colors[0],alpha=alpha_c[0],label = 'GP result (present data)')
    blue_patch = mpatches.Patch(color=plot_colors[1],alpha=alpha_c[1],label = 'With mock DESI data')
    #blue_patch = mpatches.Patch(color =plot_colors[2],alpha=alpha_c[2],label = r'DESI from $\Lambda$CDM')
    plt.subplot(gs[0])
    plt.legend(handles = [green_patch,blue_patch],loc='upper left',frameon=False)

    plt.savefig(args.output + '.' + args.plot_format)
    plt.close()



if __name__ == '__main__':
    main()
