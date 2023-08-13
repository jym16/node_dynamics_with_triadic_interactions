Modelling Higher-Order Network Dynamics in the Presence of Triadic Interactions
===============================================================================

Overview
--------
This repository contains the code for the project "Modelling Higher-Order Network Dynamics in the Presence of Triadic Interactions" by Jun Yamamoto. 
The project was submitted as a part of the requirements for the degree of MSc in Mathematics at Queen Mary University of London.
The project was supervised by Prof. Dr. Ginestra Bianconi.

The project consists of the numerical simulations of node dynamics on networks with triadic interactions, and the analysis of the results.
The codes were written in Python 3 and the following packages are required:

- `matplotlib <https://matplotlib.org>`_
- `numpy <https://numpy.org>`_
- `scipy <https://scipy.org>`_
- `sdeint <https://github.com/mattja/sdeint/>`_

The code is released under the MIT license.

Class
-----
``triadic_interactions.model`` : 

  ``NDwTIs`` : Class for numerical simulations of node dynamics on networks with triadic interactions.

Functions
---------
``triadic_interactions.computation`` :

  ``create_node_edge_incidence_matrix(edge_list)`` : Create a node-edge incidence matrix B from a given edge list.

  ``extract_by_std(X, std=3.0)`` : Extract the data within a given number of standard deviations from its mean.

  ``freedman_diaconis_rule(data, power=1. / 3., factor=2., trim=1)`` : Compute the number of bins using the Freedman-Diaconis rule.

  ``discretise(X, n_bins='fd')`` : Discretise the time series data.

  ``estimate_pdf(data, bins='fd', method='hist')`` : Estimate the probability density function.
  
  ``estimate_pdf_joint(data, bins='fd', method='hist')`` : Estimate the joint probability density function.
  
  ``estimate_pdf_conditional(data, data_cond, val_cond, bins='fd', method='hist')`` : Estimate the conditional probability density function.
  
  ``pdf_evolution(X, t_max, n_x_resolution=50)`` : Compute the evolution of the probability density function.
  
  ``covariance(data)`` : Compute the covariance.
  
  ``conditional_expectation(X, Y, Z, bins='fd')`` : Compute the conditional expectation.
  
  ``conditional_variance(X, Z, bins='fd')`` : Compute the conditional variance.
  
  ``conditional_covariance(X, Y, Z, bins='fd')`` : Compute the conditional covariance.
  
  ``conditional_correlation(X, Y, Z, bins='fd', method='default')`` : Compute the conditional correlation.
  
  ``entropy(pdf, x)`` : Compute the entropy.
  
  ``entropy_joint(pdf_joint, x)`` : Compute the joint entropy.
  
  ``conditional_mutual_information(X, Y, Z, bins='fd', method='hist')`` : Compute the conditional mutual information.

``triadic_interactions.computation`` : 

  ``plot_timeseries(X, output_file, t_max, n_samples=1, separate=False, theory=None)`` : Plot the timeseries.

  ``plot_pdf(probs, bins, output_file, f_theory=None, logscale=False, parallel=False)`` : Plot the probability distributions.

  ``plot_covariance(cov, output_file, theory=None)`` : Plot the covariance matrix.

  ``plot_conditional_expectation(Xgrids, cond_exps, stds, orders, output_file=None, theory=None)`` : Plot the conditional expectation.

  ``plot_conditional_correlation(Xgrids, cond_corr, order, output_file, std=False, Xrange=None, theory=None, f_supplement=None, threshold=None)`` : Plot the conditional correlation.

  ``plot_conditional_mutual_information(Xgrids, cmi, order, output_file, std=False, theory=None)`` : Plot conditional mutual information.

  ``visualise_evolution(evolution_data, x_grid, time_grid, output_file)`` : Visualise the time evolution of the probability density function.

Examples
--------
  The following example demonstrates how to use the ``NDwTIs`` class to simulate the node dynamics on networks with triadic interactions.
.. code block::
    from triadic_interactions.model import NDwTIs, create_node_edge_incidence_matrix
    # Node
    n_nodes = 3
    # Edge list
    edge_list = [
        [2, 3]
    ]
    n_edges = len(edge_list)
    # Incidence matrix for the structural network
    B = create_node_edge_incidence_matrix(
        edge_list
    )
    # Incidence matrix for the triadic interactions
    K = np.array([[1, 0, 0]])
    model = NDwTIs(
        B=B, K=K, w_pos=2., w_neg=1., 
        threshold=1e-3, alpha=.1, noise_std=1e-3,
        x_init=np.zeros(n_nodes), dt=1e-2, t_max=100.
    )
    # Run the simulation
    model.run()
    

Acknowledgements
----------------
The author would like to thank Prof. Dr. Ginestra Bianconi for her guidance and support throughout the project.

The author also received assistance from Dr. Anthony Baptista in implementing the ``create_node_edge_incidence_matrix`` function in ``triadic_interaction.computation`` and the ``NDwTIs`` class in ``triadic_interaction.model``.

