install
=======

Programs can be run directly from the top-level directory without needing to set `PYTHONPATH` as long as you have the required packages already installed, e.g.::

	git clone ... gphist
	cd gphist
	./infer.py --help

Required Packages
-----------------

This package was developed under python 2.7. Please file an issue (or, even better, a PR) if python 3.x support is needed.

The following python packages are required by this package:

* numpy (linalg,random)
* scipy (interpolate,stats)
* matplotlib (pyplot)
* astropy (constants,units,cosmology)

The recommended way to obtain these packages is to install a recent `anaconda <https://store.continuum.io/cshop/anaconda/>`_ distribution.

To create a suitable minimal conda environment, use::

	conda create -n gphist python=2.7 numpy scipy astropy matplotlib
