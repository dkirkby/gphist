examples
========

Quick Demo
----------

Run inference with fixed hyperparameters for quick demonstration::

	./infer.py --hyper-h 0.1 --hyper-sigma 0.02 --num-samples 100000 --output demo

Plot the results with only a CMB posterior applied::

	./plot.py --input demo.0 --posterior CMB --show --dark-energy

Full Calculation
----------------

Run inferences on a grid of hyperparameter values::

	~/multi/multi --nohup --split 0:400:20 --run "./infer.py --seed NNN --hyper-num-h 20 --hyper-num-sigma 20 --hyper-index NNN --hyper-count 20 --output hyper20_NNN --num-samples 5000000"

Combine inferences to marginalize over hyperparameters::

	./combine.py --input 'hyper20_*' --output combined20

Generate plots::

	./plot.py --input combined20 --nlp --full --zoom --output plots20/
