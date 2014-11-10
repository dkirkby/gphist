Examples
========

Run inference on a grid of hyperparameter values::

	~/multi/multi --nohup --split 0:400:20 --run "./infer.py --seed NNN --hyper-num-h 20 --hyper-num-sigma 20 --hyper-index NNN --hyper-count 20 --output hyper20_NNN --num-samples 5000000"

Generate plots::

	./plot.py --input 'hyper20_*.npz' --nlp --full --zoom --output plots/hyper- --no-examples
