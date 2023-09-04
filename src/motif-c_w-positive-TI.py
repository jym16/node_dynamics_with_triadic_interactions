# Import necessary modules
import os
import numpy as np
from triadic_interaction import *

# If the data directory does not exist
if os.path.exists("./data") == False:
    os.mkdir("./data")

# If the figure directory does not exist
if os.path.exists("./figure") == False:
    os.mkdir("./figure")

def main():
    # Set seed
    seed = 12345678
    np.random.seed(seed)

    # Define the identifier
    identifier = "motif-c_w-positive-TI"
    _bins = 50 # 'fd'

    """Base path."""
    data_basepath = "./data/" + identifier
    fig_basepath = "./figure/" + identifier

    """Data filepaths."""
    data_timeseries = os.path.join(data_basepath, "{}_timeseries.npy".format(identifier))
    data_cond_corr_123 = os.path.join(data_basepath, "{}_cond_corr_123_{}bins.npy".format(identifier, _bins))
    data_cond_corr_123_x = os.path.join(data_basepath, "{}_cond_corr_123_x_{}bins.npy".format(identifier, _bins))
    data_cond_corr_123_stderr = os.path.join(data_basepath, "{}_cond_corr_123_stderr_{}bins.npy".format(identifier, _bins))
    data_cond_corr_132 = os.path.join(data_basepath, "{}_cond_corr_132_{}bins.npy".format(identifier, _bins))
    data_cond_corr_132_x = os.path.join(data_basepath, "{}_cond_corr_132_x_{}bins.npy".format(identifier, _bins))
    data_cond_corr_132_stderr = os.path.join(data_basepath, "{}_cond_corr_132_stderr_{}bins.npy".format(identifier, _bins))
    data_cond_corr_231 = os.path.join(data_basepath, "{}_cond_corr_231_{}bins.npy".format(identifier, _bins))
    data_cond_corr_231_x = os.path.join(data_basepath, "{}_cond_corr_231_x_{}bins.npy".format(identifier, _bins))
    data_cond_corr_231_stderr = os.path.join(data_basepath, "{}_cond_corr_231_stderr_{}bins.npy".format(identifier, _bins))
    data_cmi_123 = os.path.join(data_basepath, "{}_cmi_123_{}bins.npy".format(identifier, _bins))
    data_cmi_123_x = os.path.join(data_basepath, "{}_cmi_123_x_{}bins.npy".format(identifier, _bins))
    data_cmi_132 = os.path.join(data_basepath, "{}_cmi_132_{}bins.npy".format(identifier, _bins))
    data_cmi_132_x = os.path.join(data_basepath, "{}_cmi_132_x_{}bins.npy".format(identifier, _bins))
    data_cmi_231 = os.path.join(data_basepath, "{}_cmi_231_{}bins.npy".format(identifier, _bins))
    data_cmi_231_x = os.path.join(data_basepath, "{}_cmi_231_x_{}bins.npy".format(identifier, _bins))

    """Figure filepaths."""
    fig_timeseries = os.path.join(fig_basepath, "{}_timeseries.pdf".format(identifier))
    fig_px = os.path.join(fig_basepath, "{}_probability_distribution_{}bins.pdf".format(identifier, _bins))
    fig_px_log = os.path.join(fig_basepath, "{}_probability_distribution_log_{}bins.pdf".format(identifier, _bins))
    fig_cov = os.path.join(fig_basepath, "{}_covariance.pdf".format(identifier))
    fig_cond_corr = os.path.join(fig_basepath, "{}_cond_corr_{}bins.pdf".format(identifier, _bins))
    fig_cond_corr_stderr = os.path.join(fig_basepath, "{}_cond_corr_stderr_{}bins.pdf".format(identifier, _bins))
    fig_cmi = os.path.join(fig_basepath, "{}_cmi_{}bins.pdf".format(identifier, _bins))

    # If the data directory does not exist
    if os.path.exists(data_basepath) == False:
        os.mkdir(data_basepath)
    
    # If the figure directory does not exist
    if os.path.exists(fig_basepath) == False:
        os.mkdir(fig_basepath)

    """The structural network."""
    # Number of nodes
    n_nodes = 3
    # Number of edges
    n_edges = 3
    # Edge list
    edge_list = [
        [1, 2],
        [1, 3],
        [2, 3]
    ]
    # Incidence matrix    
    B = create_node_edge_incidence_matrix(
        edge_list
    )

    """The triadic interactions."""
    # Incidence matrix of triadic interactions
    K = np.zeros((n_edges, n_nodes), dtype=np.int_)
    K[2, 0] = 1

    """The model parameters."""
    w_pos = 2.0
    w_neg = 1.0
    threshold = 0.0e-1
    alpha = 1.0e0
    noise_std = 1.0e-2
    dt = 1.0e-2
    t_max = 1.0e2
    n_timesteps = int(1 + t_max / dt)

    """The initial conditions."""
    x_init = np.zeros(n_nodes)

    """The simulation parameters."""
    n_samples = 1000
    print_frequency = n_samples // 20
    n_folds = 2 # Use 2-fold for stationary distribution
    t0 = -n_timesteps // n_folds # take the second half of the timeseries

    """The simulation."""
    # Check if the data exists
    if os.path.exists(data_timeseries):
        # Load the data
        X = np.load(data_timeseries)
    
    # If the data does not exist
    else:
        # Initialize the data matrices
        X = np.zeros(
            (n_nodes, int(t_max / dt) + 1, n_samples)
        )

        # Loop over trials
        for i in range(n_samples):
            # Construct the model
            model = NDwTIs(
                B=B, K=K, w_pos=w_pos, w_neg=w_neg, 
                threshold=threshold, alpha=alpha, noise_std=noise_std,
                x_init=x_init, dt=dt, t_max=t_max,
                external_force=None # external_force
            )
            # Run the model
            time_series = model.run()
            # Save the data
            X[:, :, i] = time_series
            # Print the progress
            if (i + 1) % print_frequency == 0:
                print(
                    "sample # : {:5d} / {:5d}".format(i + 1, n_samples)
                )
        
        # Save the data
        np.save(data_timeseries, X)

    # Plot the output
    plot_timeseries(
        X, 
        output_file=fig_timeseries, 
        t_max=t_max, 
        n_samples=3, 
        separate=True
    )

    # Trim the timeseries (for stationary distribution)
    X = X[:, t0:, :]

    # Plot the probability distribution
    def prob_theory(x, Gamma, w, a):
        return np.sqrt(a * (a + 3 * w) / (np.pi * Gamma**2 * (a + w))) * np.exp(- a * (a + 3 * w) * x**2 / (Gamma **2 * (a + w)))

    f_theory = [
        lambda x : prob_theory(x, noise_std, w_neg, alpha),
        lambda x : prob_theory(x, noise_std, w_neg, alpha),
        lambda x : prob_theory(x, noise_std, w_neg, alpha)
    ]

    # Compute the probability distributions
    pdf_X, x_bins = estimate_pdf(X[0].flatten(), bins=_bins)
    pdf_Y, y_bins = estimate_pdf(X[1].flatten(), bins=_bins)
    pdf_Z, z_bins = estimate_pdf(X[2].flatten(), bins=_bins)

    # Plot the probability distributions
    plot_pdf(
        probs=[pdf_X, pdf_Y, pdf_Z],
        bins=[x_bins, y_bins, z_bins],
        f_theory=f_theory, 
        output_file=fig_px, 
        parallel=True
    )
    
    # Compute the theoretical covariance
    cov_theo =  noise_std**2 / (2 * alpha * (alpha + 3 * w_neg)) \
        * np.array([
            alpha + w_neg, w_neg, w_neg,
            w_neg, alpha + w_neg, w_neg,
            w_neg, w_neg, alpha + w_neg
    ])

    # Compute the covariance
    cov = covariance(X)

    plot_covariance(
        cov, 
        output_file=fig_cov,
        theory=cov_theo
    )

    plot_range = [extract_by_std(X[i]) for i in range(n_nodes)]

    # Compute the theoretical conditional correlation
    cond_corr_theory = [
        lambda x : w_neg / (alpha + 2 * w_neg) + 0. * x,
        lambda x : w_neg / (alpha + 2 * w_neg) + 0. * x,
        lambda x : w_neg / (alpha + 2 * w_neg) + 0. * x,
    ]

    if os.path.exists(data_cond_corr_123) and os.path.exists(data_cond_corr_123_x) and os.path.exists(data_cond_corr_123_stderr):
        C12_3 = np.load(data_cond_corr_123)
        Xgrid12_3 = np.load(data_cond_corr_123_x)
        dC12_3 = np.load(data_cond_corr_123_stderr)
    else:
        # Compute the conditional correlation
        C12_3, Xgrid12_3, dC12_3 = conditional_correlation(
            X=X[0].flatten(),
            Y=X[1].flatten(),
            Z=X[2].flatten(),
            bins=_bins
        )
        np.save(
            data_cond_corr_123, 
            C12_3
        )
        np.save(
            data_cond_corr_123_x, 
            Xgrid12_3
        )
        np.save(
            data_cond_corr_123_stderr,
            dC12_3
        )
    if os.path.exists(data_cond_corr_132) and os.path.exists(data_cond_corr_132_x) and os.path.exists(data_cond_corr_132_stderr):
        C13_2 = np.load(data_cond_corr_132)
        Xgrid13_2 = np.load(data_cond_corr_132_x)
        dC13_2 = np.load(data_cond_corr_132_stderr)
    else:
        C13_2, Xgrid13_2, dC13_2 = conditional_correlation(
            X=X[0].flatten(),
            Y=X[2].flatten(),
            Z=X[1].flatten(),
            bins=_bins
        )
        np.save(
            data_cond_corr_132, 
            C13_2
        )
        np.save(
            data_cond_corr_132_x, 
            Xgrid13_2
        )
        np.save(
            data_cond_corr_132_stderr,
            dC13_2
        )
    if os.path.exists(data_cond_corr_231) and os.path.exists(data_cond_corr_231_x) and os.path.exists(data_cond_corr_231_stderr):
        C23_1 = np.load(data_cond_corr_231)
        Xgrid23_1 = np.load(data_cond_corr_231_x)
        dC23_1 = np.load(data_cond_corr_231_stderr)
    else:
        C23_1, Xgrid23_1, dC23_1 = conditional_correlation(
            X=X[1].flatten(),
            Y=X[2].flatten(),
            Z=X[0].flatten(), 
            bins=_bins
        )
        np.save(
            data_cond_corr_231, 
            C23_1
        )
        np.save(
            data_cond_corr_231_x, 
            Xgrid23_1
        )
        np.save(
            data_cond_corr_231_stderr,
            dC23_1
        )

    # Plot the conditional correlation
    plot_conditional_correlation(
        Xgrids=[Xgrid23_1, Xgrid13_2, Xgrid12_3], 
        cond_corr=[C23_1, C13_2, C12_3],
        order=[(2,3,1), (1,3,2), (1,2,3)],
        output_file=fig_cond_corr, 
        std=False,
        Xrange=plot_range,
        theory=cond_corr_theory
    )

    # Plot the conditional correlation with standard errors
    plot_conditional_correlation(
        Xgrids=[Xgrid23_1, Xgrid13_2, Xgrid12_3], 
        cond_corr=[C23_1, C13_2, C12_3],
        order=[(2,3,1), (1,3,2), (1,2,3)],
        output_file=fig_cond_corr_stderr, 
        std=[dC23_1, dC13_2, dC12_3],
        Xrange=plot_range,
        theory=cond_corr_theory
    )

    # Compute the theoretical conditional mutual information
    cmi_theory = [
        lambda x : 0.5 * np.log((alpha + 2 * w_neg) ** 2 / ((alpha + w_neg) * (alpha + 3 * w_neg))) + 0. * x,
        lambda x : 0.5 * np.log((alpha + 2 * w_neg) ** 2 / ((alpha + w_neg) * (alpha + 3 * w_neg))) + 0. * x,
        lambda x : 0.5 * np.log((alpha + 2 * w_neg) ** 2 / ((alpha + w_neg) * (alpha + 3 * w_neg))) + 0. * x
    ]

    if os.path.exists(data_cmi_231) and os.path.exists(data_cmi_231_x):
        CMI23_1 = np.load(data_cmi_231)
        Xgrid23_1 = np.load(data_cmi_231_x)
    else:
        CMI23_1, _, Xgrid23_1 = conditional_mutual_information(
            X=X[1].flatten(),
            Y=X[2].flatten(),
            Z=X[0].flatten(), 
            bins=_bins,
            method='kde'
        )
        np.save(
            data_cmi_231, 
            CMI23_1
        )
        np.save(
            data_cmi_231_x, 
            Xgrid23_1
        )
    if os.path.exists(data_cmi_132) and os.path.exists(data_cmi_132_x):
        CMI13_2 = np.load(data_cmi_132)
        Xgrid13_2 = np.load(data_cmi_132_x)
    else:
        CMI13_2, _, Xgrid13_2 = conditional_mutual_information(
            X=X[0].flatten(),
            Y=X[2].flatten(),
            Z=X[1].flatten(), 
            bins=_bins,
            method='kde'
        )
        np.save(
            data_cmi_132, 
            CMI13_2
        )
        np.save(
            data_cmi_132_x, 
            Xgrid13_2
        )
    if os.path.exists(data_cmi_123) and os.path.exists(data_cmi_123_x):
        CMI12_3 = np.load(data_cmi_123)
        Xgrid12_3 = np.load(data_cmi_123_x)
    else:
        CMI12_3, _, Xgrid12_3 = conditional_mutual_information(
            X=X[0].flatten(),
            Y=X[1].flatten(),
            Z=X[2].flatten(), 
            bins=_bins,
            method='kde'
        )
        np.save(
            data_cmi_123, 
            CMI12_3
        )
        np.save(
            data_cmi_123_x, 
            Xgrid12_3
        )
    
    # Plot the conditional mutual information
    plot_conditional_mutual_information(
        Xgrids=[Xgrid23_1, Xgrid13_2, Xgrid12_3], 
        cmi=[CMI23_1, CMI13_2, CMI12_3],
        order=[(2,3,1), (1,3,2), (1,2,3)],
        output_file=fig_cmi,
        Xrange=plot_range,
        theory=cmi_theory
    )

if __name__ == "__main__":
    main()

"""End of file"""