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
    seed = 123456789
    np.random.seed(seed)

    # Define the identifier
    identifier = "C1a+"
    _bins = 50 # 'fd' if you want to use Freedman-Diaconis rule

    """Base paths."""
    data_basepath = "./data/testcase" + identifier
    fig_basepath = "./figure/testcase" + identifier

    """Data filepaths."""
    data_timeseries = os.path.join(data_basepath, "{}_timeseries.npy".format(identifier))
    data_p_evolution = os.path.join(data_basepath, "{}_p_evolution.npy".format(identifier))
    data_cond_corr_123 = os.path.join(data_basepath, "{}_cond_corr_123_{}bins.npy".format(identifier, _bins))
    data_cond_corr_123_x = os.path.join(data_basepath, "{}_cond_corr_123_x_{}bins.npy".format(identifier, _bins))
    data_cond_corr_123_stderr = os.path.join(data_basepath, "{}_cond_corr_123_stderr_{}bins.npy".format(identifier, _bins))
    data_cond_corr_132 = os.path.join(data_basepath, "{}_cond_corr_132_{}bins.npy".format(identifier, _bins))
    data_cond_corr_132_x = os.path.join(data_basepath, "{}_cond_corr_132_x_{}bins.npy".format(identifier, _bins))
    data_cond_corr_132_stderr = os.path.join(data_basepath, "{}_cond_corr_132_stderr_{}bins.npy".format(identifier, _bins))
    data_cond_corr_231 = os.path.join(data_basepath, "{}_cond_corr_231_{}bins.npy".format(identifier, _bins))
    data_cond_corr_231_x = os.path.join(data_basepath, "{}_cond_corr_231_x_{}bins.npy".format(identifier, _bins))
    data_cond_corr_231_stderr = os.path.join(data_basepath, "{}_cond_corr_231_stderr_{}bins.npy".format(identifier, _bins))

    """Figure filepaths."""
    fig_timeseries = os.path.join(fig_basepath, "{}_timeseries.pdf".format(identifier))
    fig_px = os.path.join(fig_basepath, "{}_probability_distribution_{}bins.pdf".format(identifier, _bins))
    fig_px_log = os.path.join(fig_basepath, "{}_probability_distribution_log_{}bins.pdf".format(identifier, _bins))
    fig_p_evolution = os.path.join(fig_basepath, "{}_p_evolution.pdf".format(identifier))
    fig_cov = os.path.join(fig_basepath, "{}_covariance.pdf".format(identifier))
    fig_cond_corr = os.path.join(fig_basepath, "{}_cond_corr_{}bins.pdf".format(identifier, _bins))
    fig_cond_corr_stderr = os.path.join(fig_basepath, "{}_cond_corr_stderr_{}bins.pdf".format(identifier, _bins))

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
    n_edges = 1
    # Edge list
    edge_list = [
        [2, 3]
    ]
    # Incidence matrix    
    B = create_node_edge_incidence_matrix(
        edge_list
    )

    """The triadic interactions."""
    # Incidence matrix of triadic interactions
    K = np.zeros((n_edges, n_nodes), dtype=np.int_)
    K[0, 0] = 1

    """The model parameters."""
    w_pos = 2.0
    w_neg = 1.0
    threshold = -1.0e-1
    alpha = 1.0e-1
    noise_std = 1.0e-1
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

    n_x_resolution = 100
    X_max_amp = np.max(np.abs(X)) * 1.5
    X_bins = np.linspace(-X_max_amp, X_max_amp, n_x_resolution+1)
    time_grid = np.linspace(0, t_max, int(t_max / dt + 1))

    time_evolution = np.zeros((n_nodes, n_x_resolution, n_timesteps))
    for i in range(n_nodes):
        for t in range(n_timesteps):
            p_t, _ = estimate_pdf(X[i, t, :], bins=X_bins)
            assert(1 - np.sum(p_t) < 1e-10)
            time_evolution[i, :, t] = p_t
    np.save(data_p_evolution, time_evolution)

    visualise_evolution(time_evolution, X_bins, time_grid, fig_p_evolution)

    # Trim the timeseries (for stationary distribution)
    X = X[:, t0:, :]

    # Plot the probability distribution
    def prob_theory1(x, Gamma, w, a):
        return np.sqrt(a / (np.pi * Gamma**2)) * np.exp(- a * x**2 / Gamma **2)

    def prob_theory23(x, Gamma, w, a):
        return np.sqrt(a * (a + 2. * w) / (np.pi * Gamma**2 * (a + w))) * np.exp(- a * (a + 2. * w) * x**2 / (Gamma **2. * (a + w)))

    f_theory = [
        lambda x : prob_theory1(x, noise_std, w_neg, alpha),
        lambda x : prob_theory23(x, noise_std, w_neg, alpha),
        lambda x : prob_theory23(x, noise_std, w_neg, alpha)
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

    plot_pdf(
        probs=[pdf_X, pdf_Y, pdf_Z],
        bins=[x_bins, y_bins, z_bins],
        f_theory=f_theory, 
        output_file=fig_px, 
        logscale=True,
        parallel=True
    )

    # Compute the theoretical covariance
    cov_theo =  noise_std**2 / (2 * alpha * (alpha + 2 * w_neg)) \
        * np.array([
            alpha + 2 * w_neg, 0., 0., 
            0., alpha + w_neg, w_neg, 
            0, w_neg, alpha + w_neg       
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
        lambda x : w_neg / (alpha + w_neg) + 0. * x,
        lambda x : 0. + 0. * x,
        lambda x : 0. + 0. * x
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

    # Plot the conditional correlation for X_2, X_3 | X_1
    fig_cond_corr_231 = os.path.join(fig_basepath, "{}_cond_corr_all-samples_{}bins_rho231.pdf".format(identifier, _bins))

    plot_conditional_correlation(
        Xgrids=Xgrid23_1, 
        cond_corr=C23_1,
        order=(2,3,1),
        output_file=fig_cond_corr_231, 
        std=False,
        theory=cond_corr_theory[0]
    )

if __name__ == "__main__":
    main()

"""End of file"""