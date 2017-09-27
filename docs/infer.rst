infer
=====

Infer the cosmological expansion history using a Gaussian process prior.

Memory Usage
------------

Gaussian process realizations = (8+8+4)*num_evol_samples*num_steps
where 8+8+4 combines DH(8) and DA(8) samples and the calculated bin_indices(4).

Histograms = (8+8)*num_bins*num_hist_steps*2**npost
where 8+8 combines DH(8) and DA(8) histograms.

The histograms are written to disk, so this is also the output file size.
