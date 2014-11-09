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

~/multi/multi --nohup --split 0:400:16 --run "./infer.py --seed NNN --hyper-num-h 20 --hyper-num-sigma 20 --hyper-index NNN --hyper-count 16 --output hyper20_NNN --num-samples 5000000"

./plot.py --input 'hyper_*.npz' --nlp --full --zoom --output plots/hyper- --no-examples
