"""
Triadic Interactions Package.

This package contains a Node Dynamics with Triadic Interaction class and a set of functions for computations and visualization.
"""
from triadic_interaction.model import NDwTIs
from triadic_interaction.computation import (
    create_node_edge_incidence_matrix,
    extract_by_std,
    freedman_diaconis_rule,
    estimate_pdf,
    estimate_pdf_joint,
    estimate_pmf,
    estimate_pmf_joint,
    estimate_mutual_information,
    covariance,
    conditional_correlation,
    conditional_mutual_information
)
from triadic_interaction.visualization import (
    plot_timeseries,
    plot_pdf,
    plot_covariance,
    plot_conditional_correlation,
    plot_conditional_mutual_information
)

__all__ = [
    ### Model ###
    'NDwTIs',

    ### Computation ###
    'create_node_edge_incidence_matrix',
    'extract_by_std',
    'freedman_diaconis_rule',
    'estimate_pdf',
    'estimate_pdf_joint',
    'estimate_pmf',
    'estimate_pmf_joint',
    'estimate_mutual_information',
    'covariance',
    'conditional_correlation',
    'conditional_mutual_information',

    ### Visualization ###
    'plot_timeseries',
    'plot_pdf',
    'plot_covariance',
    'plot_conditional_correlation',
    'plot_conditional_mutual_information',
]

"""End of file."""