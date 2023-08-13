"""
Triadic Interactions Package.

This package contains a Node Dynamics with Triadic Interaction class and a set of functions for computations and visualization.
"""
from triadic_interaction.model import NDwTIs
from triadic_interaction.computation import (
    create_node_edge_incidence_matrix,
    extract_by_std,
    pdf_evolution,
    freedman_diaconis_rule,
    discretise,
    estimate_pdf,
    estimate_pdf_joint,
    estimate_pdf_conditional,
    entropy,
    entropy_joint,
    conditional_mutual_information,
    covariance,
    conditional_expectation,
    conditional_variance,
    conditional_covariance,
    conditional_correlation
)
from triadic_interaction.visualization import (
    plot_timeseries,
    plot_pdf,
    plot_covariance,
    plot_conditional_expectation,
    plot_conditional_correlation,
    plot_conditional_mutual_information,
    visualise_evolution
)

__all__ = [
    ### Model ###
    'NDwTIs',

    ### Computation ###
    'create_node_edge_incidence_matrix',
    'extract_by_std',
    'pdf_evolution',
    'freedman_diaconis_rule',
    'discretise',
    'estimate_pdf',
    'estimate_pdf_joint',
    'estimate_pdf_conditional',
    'entropy',
    'entropy_joint',
    'conditional_mutual_information',
    'covariance',
    'conditional_expectation',
    'conditional_variance',
    'conditional_covariance',
    'conditional_correlation',

    ### Visualization ###
    'plot_timeseries',
    'plot_pdf',
    'plot_covariance',
    'plot_conditional_expectation',
    'plot_conditional_correlation',
    'plot_conditional_mutual_information',
    'visualise_evolution',
]

"""End of file."""