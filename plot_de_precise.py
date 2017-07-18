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



    # Do we have anything to plot?
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
    loaded = np.load('lya_test5.npz')
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



    # Load the input file.
    #loaded_DESI = np.load('DESI_phi_test2.npz')
    loaded_bias = np.load('bias_phi_de.npz')
    
    DH_hist_bias = loaded_bias['DH_hist']
    DA_hist_bias = loaded_bias['DA_hist']
    de_hist_bias = loaded_bias['de_hist']
    f_hist_bias = loaded_bias['phi_hist']
    zvalues_bias = loaded_bias['zvalues']
    #print zvalues_DESI.shape
    #print zvalues_bias.shape
    #assert zvalues ==zvalues_DESI , 'the zvalues are different'
    lna=-np.log(1 + zvalues[::-1])
    bin_range = loaded['bin_range']

    # Initialize the posterior permutations.
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
    # Initialize the hyperparameter grid.

    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.size': 40})
    name = 'idk'
    #matplotlib.rcParams.update({'font.size': 22})
    fig = plt.figure(name,figsize=(12,8))
    fig.subplots_adjust(left=0.15,bottom=0.14,right=0.97,
                        top=0.98,wspace=0.0,hspace=0.18)
    fig.set_facecolor('white')

    #plot_colors = ('red','blue','green')
    plot_colors = [ np.array((55,126,184))/255., np.array((152,78,163))/255., np.array((255,128,0))/255.]
    #plot_c = ('r','b','g')
    alpha_c = (0.7,0.6,0.6)
    gs = gridspec.GridSpec(1,1)




    for iperm,j in zip((31,),(0,)):
        print posterior_names[perms[iperm]]
        ide = 0
        # Plot dark-energy evolution up to zmax.

        iend = 1+np.argmax(zvalues > args.zmax)
        de_labels = (r'$\omega_{\Lambda}(z)/h_0^2$',r'$\omega_{\phi}(z)/h(z)^2$')
        assert len(de_labels) == len(de_hist), 'Mismatch in number of dark-energy variables'
        # Bin range for DE histograms is hardcoded for now.
        de_bin_range = np.array([-10.,+10.])
       
        de_limits = gphist.analysis.calculate_confidence_limits(
                    de_hist[ide,iperm,:iend+1],[args.level],de_bin_range)
        interp_de_low = interp1d(zvalues[:iend+1],de_limits[0,:iend+1],kind='cubic')
        interp_de_mid = interp1d(zvalues[:iend+1],de_limits[1,:iend+1],kind='cubic')
        interp_de_high = interp1d(zvalues[:iend+1],de_limits[2,:iend+1],kind='cubic')

        plt.subplot(gs[0])
        plt.xscale('linear')
        #plt.grid(True)
        plt.xlim([0.,args.zmax])
        #plt.fill_between(zvalues_DESI[:iend+1],de_limits[0],de_limits[-1],
        #            facecolor=plot_c[j],alpha=alpha_c[j])
        plt.fill_between(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_de_low(np.linspace(0,zvalues[iend],num=100,endpoint=True)),interp_de_high(np.linspace(0,zvalues[iend],num=100,endpoint=True)), color=plot_colors[j],alpha=alpha_c[j])
        #plt.plot(np.linspace(0,zvalues_DESI[iend],num=100,endpoint=True),interp_de_low(np.linspace(0,zvalues_DESI[iend],num=100,endpoint=True)),plot_c[j]+'-')
        plt.plot(np.linspace(0,zvalues[iend],num=100,endpoint=True),interp_de_mid(np.linspace(0,zvalues[iend],num=100,endpoint=True)),color=plot_colors[j],linestyle='-',linewidth=4.0)
        #plt.plot(np.linspace(0,zvalues_DESI[iend],num=100,endpoint=True),interp_de_high(np.linspace(0,zvalues_DESI[iend],num=100,endpoint=True)),plot_c[j]+'-')
        #plt.plot(zvalues_DESI[:iend],de_limits[0],plot_c[j]+'-')
        #plt.plot(zvalues_DESI[:iend],de_limits[1],plot_c[j]+'-')
        #plt.plot(zvalues_DESI[:iend],de_limits[2],plot_c[j]+'-')
        plt.plot(zvalues[:iend],de0[ide,:iend],'k--',linewidth=4.0)
        plt.xlabel(r'$z$',fontsize=40)
        plt.ylim([-1.0,1.5])
        plt.ylabel(de_labels[ide],fontsize=40)

    green_patch = mpatches.Patch(color=plot_colors[0],alpha=alpha_c[0],label = 'Precise H0')
    plt.subplot(gs[0])
    plt.legend(handles = [green_patch],loc='lower left',fontsize=40,frameon=False)

    plt.savefig(args.output +'.' + args.plot_format)
    plt.close()







if __name__ == '__main__':
    main()
