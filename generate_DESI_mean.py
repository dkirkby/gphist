#!/usr/bin/env python
"""Plot expansion history inferences.
"""

import argparse

import numpy as np
# matplotlib is imported inside main()
from scipy.optimize import minimize

import gphist
from scipy import interpolate

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
    #f_hist = f_loaded['phi_hist']
    #print 'the shape of the combined phi histogram is '+str(phi_hist.shape)
    #print 'the shape of the combined DH histogram is '+str(DH_hist.shape)
    #phi_realizations = loaded['phi_realizations']
    DH0 = loaded['DH0']
    DA0 = loaded['DA0']
    de0 = loaded['de0']
    #f0 = f_loaded['phi0']
    zvalues = loaded['zvalues']
    lna=-np.log(1 + zvalues[::-1])
    fixed_options = loaded['fixed_options']
    bin_range = loaded['bin_range']
    hyper_range = loaded['hyper_range']
    posterior_names = loaded['posterior_names']
    # The -log(P) array is only present if this file was written by combine.py
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
 
    iperm =31
    name = '-'.join(posterior_names[perms[iperm]]) or 'Prior'
    
    print '%d : %s' % (iperm,name)


    
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
    new_z_values = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,
                            1.96,2.12,2.28,2.43,2.59,2.75,2.91,3.07,3.23,3.39,3.55])
    DA_interp = interpolate.interp1d(zvalues,DA_limits[1])
    DH_interp = interpolate.interp1d(zvalues,DH_limits[1])
    new_DA_val = DA_interp(new_z_values)
    new_DH_val = DH_interp(new_z_values)
    DA0_interp = interpolate.interp1d(zvalues,DA0)     
    DH0_interp = interpolate.interp1d(zvalues,DH0)
    new_DA_lcdm = DA0_interp(new_z_values)
    new_DH_lcdm = DH0_interp(new_z_values)  
    np.savetxt('DESI_means.txt',(new_z_values,new_DH_val,new_DA_val)) 
    np.savetxt('DESI_lcdm.txt',(new_z_values,new_DH_lcdm,new_DA_lcdm))
 
 
    
if __name__ == '__main__':
    main()
