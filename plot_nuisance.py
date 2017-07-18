#!/usr/bin/env python
"""Plot expansion history inferences.
"""

import argparse

import numpy as np
# matplotlib is imported inside main()
from scipy.optimize import minimize
from scipy.interpolate import interp2d

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
    
    if not args.output and not args.show:
        print 'No output requested.'

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

    # Initialize the posterior permutations.
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
    # Initialize the hyperparameter grid.
    n_samples,n_h,n_sigma = fixed_options
    h_min,h_max,sigma_min,sigma_max = hyper_range
    hyper_grid = gphist.process.HyperParameterLogGrid(
        n_h,h_min,h_max,n_sigma,sigma_min,sigma_max)

    print h_min
    print h_max
    print n_h
    print sigma_min
    print sigma_max
    print n_sigma
    # Initialize -log(P) plotting.
    if args.nlp:
        # Factor of 0.5 since -logP = 0.5*chisq
        nlp_levels2 = 0.5*gphist.analysis.get_delta_chisq(num_dof=2)
        nlp_levels = [0,nlp_levels2[0],nlp_levels2[1]]
        print nlp_levels

    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams.update({'font.size': 24})
    name = 'idk'
    fig = plt.figure(name,figsize=(8,6))
    fig.subplots_adjust(left=0.13,bottom=0.12,right=0.97,
                        top=0.99)
    fig.set_facecolor('white')
    # Find first z index beyond zmax.
    #                   G                             r                                 b
    plot_colors = [np.array((77,175,74))/255., np.array((228,26,28))/255.,  np.array((55,126,184))/255.,]
    #plot_colors = ['Greens','Reds','Blues']

    # Loop over posterior permutations.
    for i,j in zip((3,28,31),(0,1,2)):
    #for i,j in zip((31,),(0,)):
        print posterior_names[perms[i]]
        if args.nlp:
            plt.xscale('log')
            plt.yscale('log')
            missing = hyper_nlp[i] == 0
            nlp_min = np.min(hyper_nlp[i,np.logical_not(missing)])
            nlp = np.ma.array(hyper_nlp[i]-nlp_min,mask=missing)
            print hyper_grid.sigma_edges
            print hyper_grid.h_edges
            print nlp.shape
            sigma_old = np.logspace(-3,0,16,endpoint=True)
            h_old = np.logspace(-2,np.log10(0.2),16,endpoint=True)
            nlp_interp = interp2d(sigma_old,h_old,nlp)
            #plt.pcolormesh(hyper_grid.sigma_edges,hyper_grid.h_edges,nlp,
            #    cmap='rainbow',vmin=0.,vmax=50.,rasterized=True)
            #plt.colorbar() plot_colors[j
            #plt.contourf(hyper_grid.sigma,hyper_grid.h,nlp,cmap = 'viridis',
            #        levels=nlp_levels,alpha=0.7,linestyles=('-','--',':'))
            #plt.pcolormesh(hyper_grid.sigma_edges,hyper_grid.h_edges,nlp,
            #    cmap='viridis',vmin=0.,vmax=7.,rasterized=True)
            sigma_new = np.logspace(-3,0,256,endpoint=True)
            h_new = np.logspace(-2,np.log10(0.2),256,endpoint=True)
            plt.pcolormesh(sigma_new,h_new,nlp_interp(sigma_new,h_new),
                cmap='viridis',vmin=0.,vmax=7.,rasterized=True)            
            plt.contour(sigma_new,h_new,nlp_interp(sigma_new,h_new),colors = plot_colors[j],
                levels=nlp_levels,linewidth=(10.0,10.0),linestyles=('--','-'))        
            plt.xlabel(r'Hyperparameter $\sigma$')
            plt.ylabel(r'Hyperparameter $h$')
            
    plt.savefig(args.output + 'NLP.' + args.plot_format)
    plt.close()






if __name__ == '__main__':
    main()
