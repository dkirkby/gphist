# Cosmological Expansion History Inference using Gaussian Processes

## Overview

## Design Notes

### Memory Usage

Gaussian process realizations = (8+8+4)*num_evol_samples*num_steps
where 8+8+4 combines DH(8) and DA(8) samples and the calculated bin_indices(4).

Histograms = (8+8)*num_bins*num_hist_steps*2**npost
where 8+8 combines DH(8) and DA(8) histograms.

The histograms are written to disk, so this is also the output file size.

## Results

## Examples

Run hyperparameter marginalization:

~/multi/multi --nohup --split 0:25:1 --run "./infer.py --seed NNN --hyper-num-h 5 --hyper-num-sigma 5 --hyper-index NNN --output hyper_NNN --num-samples 1000000"

./plot.py --input 'hyper_*.npz' --nll --full --zoom --output plots/hyper-
