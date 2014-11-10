#!/usr/bin/env python
"""Combine expansion history inferences.
"""

import argparse
import glob

import numpy as np

import gphist

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',type = str, default = None,
        help = 'name of input file(s) to read (wildcard patterns are supported)')
    parser.add_argument('--output', type = str, default = None,
        help = 'name of output file to write (extension .npz will be added)')
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

    # Loop over the input files.
    random_states = { }
    for index,input_file in enumerate(input_files):
        loaded = np.load(input_file)
        print 'Loaded',input_file
        if index == 0:
            DH_hist = loaded['DH_hist']
            DA_hist = loaded['DA_hist']
            DH0 = loaded['DH0']
            DA0 = loaded['DA0']
            zvalues = loaded['zvalues']
            fixed_options = loaded['fixed_options']
            bin_range = loaded['bin_range']
            hyper_range = loaded['hyper_range']
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
        else:
            # Distance arrays might differ by roundoff errors because of different downsampling.
            assert np.allclose(DH0,loaded['DH0']),'Found inconsistent DH0'
            diff = DA0-loaded['DA0']
            print diff[1:]/DA0[1:]
            #assert np.allclose(DA0,loaded['DA0']),'Found inconsistent DA0'
            assert np.allclose(zvalues,loaded['zvalues']),'Found inconsistent zvalues'
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
        DH_realizations = loaded['DH_realizations']
        DA_realizations = loaded['DA_realizations']
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

    if args.output:
        output_name = '%s.npz' % args.output
        # Relative to the infer.py output format, we drop arrays whose binning
        # varies between inputs (*_full and *_realizations), and add hyper_nlp.
        np.savez(output_name,
            DH_hist=DH_hist,DA_hist=DA_hist,DH0=DH0,DA0=DA0,zvalues=zvalues,
            fixed_options=fixed_options,bin_range=bin_range,hyper_range=hyper_range,
            posterior_names=posterior_names,hyper_nlp=hyper_nlp)

if __name__ == '__main__':
    main()
