# Cosmological Expansion History Inference using Gaussian Processes

## Overview

## Results

## Examples

Run hyperparameter marginalization:

~/multi/multi --nohup --split 0:25:1 --run "./infer.py --seed NNN --hyper-index NNN --output hyper_NNN --num-samples 1000000 --num-cycles 10"

./plot.py --output plots/ --full --zoom --dark-energy --input 'hyper_*'