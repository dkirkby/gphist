combine
=======

The combine program reads the ouputs from a set of compatible infererences (produced by :doc:`infer`), performs some checks that they are indeed compatible, and generates a single combined output file suitable for use with the :doc:`plot` command.

Compatilibity Checks
--------------------

All inputs must use the same:
 * reference cosmology,
 * redshift values for histogramming,
 * histogram binning,
 * number of generated samples,
 * hyperparameter grid,
 * posteriors (only the names are actually checked).

In addition, inputs must be generate with different initial random states.

Output Format
-------------

The output format is a numpy archive (.npz) containing a subset of the arrays written by :doc:`infer` and adding an array of -log(P) values where P is the posterior probability.
