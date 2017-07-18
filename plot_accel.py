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
    

    

    # Initialize matplotlib.
    import matplotlib as mpl
    if not args.show:
        # Use the default backend, which does not require X11 on unix systems.
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    from matplotlib import gridspec
    from matplotlib import rc

    # Load the input file.
    loaded = np.load(args.input + '.npz')
    #f_loaded = np.load('npz_files/phi_test/test2.0.npz')
    q_hist = loaded['q_hist']
    #print 'the shape of the combined phi histogram is '+str(phi_hist.shape)
    #print 'the shape of the combined DH histogram is '+str(DH_hist.shape)
    #phi_realizations = loaded['phi_realizations']
    
    q0 = loaded['q0']
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



    #rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.family' : 'serif'})
    plt.rcParams.update({'font.size': 24})
    name = 'idk'
    fig = plt.figure(name,figsize=(12,8))
    fig.subplots_adjust(left=0.09,bottom=0.10,right=0.975,
                        top=0.99,wspace=0.0,hspace=0.18)
    fig.set_facecolor('white')
            # Find first z index beyond zmax.

    


    iend = 1+np.argmax(zvalues > args.zmax)
    #plot_colors = ('blue','green','magenta')
    #plot_c_alt = ('b', 'g','r')
    #plot_c = ('b','g','m')
    #alpha_c = (0.25,0.5)
                # red                               blue                      green
    plot_c = [np.array((228,26,28))/255., np.array((55,126,184))/255., np.array((77,175,74))/255.]
    alpha_c = (0.6,0.6,0.6)
    gs = gridspec.GridSpec(1,1)
    
    for i,j in zip((31,),(1,)):
        print posterior_names[perms[i]]
        name = '-'.join(posterior_names[perms[i]])
        q_ratio_limits = gphist.analysis.calculate_confidence_limits(
                                q_hist[i],[args.level],bin_range)
        q_limits = q_ratio_limits*q0
          
      
        #plt.subplot(gs[1])
        #plt.xscale('log')
        #plt.grid(True)
        #plt.xlim([args.zmax,33.])
        #plt.ylim([0.0,1.1])
        #plt.fill_between(zvalues,phi_limits[0,::-1]/(1+zvalues),phi_limits[-1,::-1]/(1+zvalues),facecolor=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues,phi_limits[0,::-1]/(1+zvalues),plot_c[j]+':')
        #plt.plot(zvalues,phi_limits[1,::-1]/(1+zvalues),plot_c[j]+'-')
        #plt.plot(zvalues,phi_limits[2,::-1]/(1+zvalues),plot_c[j]+':')        
        #plt.xlabel(r'$z$')
        #plt.ylabel(r'$D = \phi a$')
        #frame1 = plt.gca()
        #frame1.axes.yaxis.set_ticklabels([])
        
        
        
        
        plt.subplot(gs[0])
        plt.xscale('log')
        #plt.grid(True)
        plt.xlim([1,1e3])
        #plt.ylim([0.0,1.1])
        plt.fill_between(1+zvalues,q_limits[-1,::-1],q_limits[0,::-1],facecolor=plot_c[j],alpha=alpha_c[j])
        plt.plot(1+zvalues,q_limits[0,::-1],color=plot_c[j],linestyle='-')
        plt.plot(1+zvalues,q_limits[1,::-1],color=plot_c[j],linestyle='-',linewidth=4.0)
        plt.plot(1+zvalues,q_limits[-1,::-1],color=plot_c[j],linestyle='-')
        plt.plot(1+zvalues,q0[0,::-1],'k-',label = r'$\Lambda CDM$',linewidth=4.0)
        #plt.fill_between(zvalues,phi_limits[0,::-1]/(1+zvalues)/sigma_8,phi_limits[-1,::-1]/(1+zvalues)/sigma_8,facecolor=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(zvalues,phi_limits[0,::-1]/(1+zvalues)/sigma_8,plot_c[j]+':')
        #plt.plot(zvalues,phi_limits[1,::-1]/(1+zvalues)/sigma_8,plot_c[j]+'-')
        #plt.plot(zvalues,phi_limits[2,::-1]/(1+zvalues)/sigma_8,plot_c[j]+':')
        #plt.plot(zvalues,phi0[0,::-1]/(1+zvalues)/sigma_8,'k-',label = r'$\Lambda CDM$')
        plt.xlabel(r'$1+z$',fontsize=28)
        plt.ylabel(r'$1+q$',fontsize=28)
        
    #green_patch = mpatches.Patch(color=plot_colors[0],alpha=alpha_c[0],label = 'GP Result')
    green_patch = mpatches.Patch(color=plot_c[1],alpha=alpha_c[1],label = 'GP Result')
    #red_circle = mlines.Line2D([], [], color='red', marker='o', markersize=15, label='Real Space')
    #blue_square = mlines.Line2D([], [], color='green', marker='s', markersize=15, label='Harmonic Space')
    #red_patch = mpatches.Patch(color = plot_colors[2],alpha=alpha_c[2],label = 'CMB-SN-LRG')
    plt.legend(handles = [green_patch],loc='best',frameon=False)
    #plt.legend(loc='best')
    plt.savefig(args.output + '_accel' + '.' + args.plot_format)
    plt.close()











if __name__ == '__main__':
    main()
