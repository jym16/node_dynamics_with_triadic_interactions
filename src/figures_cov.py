from triadic_interaction import *
import numpy as np
import os 
import matplotlib.pyplot as plt

def main():
    # Define the identifier
    identifiers = [
        "motif-a_w-positive-TI", "motif-a_w-negative-TI", 
        "motif-b_w-positive-TI", "motif-b_w-negative-TI",
        "motif-c_w-positive-TI", "motif-c_w-negative-TI"
    ]
    _bins = 50 # 'fd' if you want to use Freedman-Diaconis rule
    n_motifs = 3
    n_signs = 2

    output_file = "./figure/comparison_cov.pdf"

    """Base paths."""
    data_basepaths = ["./data/testcase" + id for id in identifiers]

    """Data filepaths."""
    data_timeseries = [os.path.join(path, "{}_timeseries.npy".format(id)) for path, id in zip(data_basepaths, identifiers)]

    """The structural network."""
    # Number of nodes
    n_nodes = 3

    """The model parameters."""
    w_pos = 2.0
    w_neg = 1.0
    threshold = 1.0e-3
    alpha = 1.0e0
    noise_std = 1.0e-2
    dt = 1.0e-2
    t_max = 1.0e2
    n_timesteps = int(1 + t_max / dt)

    # Theory
    cov_theo_a = noise_std**2 / (2 * alpha * (alpha + 2 * w_neg)) \
        * np.array([
            alpha + 2 * w_neg, 0., 0., 
            0., alpha + w_neg, w_neg, 
            0, w_neg, alpha + w_neg       
    ])
    cov_theo_b =  noise_std**2 / (2. * alpha * (alpha + w_neg) * (alpha + 3. * w_neg)) \
        * np.array([
            alpha**2 + 3 * alpha * w_neg + w_neg**2, w_neg * (alpha + w_neg), w_neg**2,
            w_neg * (alpha + w_neg), (alpha + w_neg)**2, w_neg * (alpha + w_neg),
            w_neg**2, w_neg * (alpha + w_neg), alpha**2 + 3 * alpha * w_neg + w_neg**2 
    ])
    cov_theo_c =  noise_std**2 / (2 * alpha * (alpha + 3 * w_neg)) \
        * np.array([
            alpha + w_neg, w_neg, w_neg,
            w_neg, alpha + w_neg, w_neg,
            w_neg, w_neg, alpha + w_neg
    ])

    f_theory = [cov_theo_a, cov_theo_b, cov_theo_c]

    """The simulation parameters."""
    n_folds = 2 # Use 2-fold for stationary distribution
    t0 = -n_timesteps // n_folds # take the second half of the timeseries

    # Load the data and trim them
    Xs = [np.load(data_path)[:, t0:, :] for data_path in data_timeseries]

    # Compute the probability distributions
    COVS = []
    for X in Xs:
        COVS.append(covariance(X))

    LABELS = [
        '$\Sigma_{11}$', '$\Sigma_{12}$', '$\Sigma_{13}$',
        '$\Sigma_{21}$', '$\Sigma_{22}$', '$\Sigma_{23}$',
        '$\Sigma_{31}$', '$\Sigma_{32}$', '$\Sigma_{33}$'
    ]

    # matplotlib configuration
    MPL_CONFIG = {
        # Figure style
        'figure.figsize' : (3, 2.5),        # specify figure size
        # Font style
        'font.size' : 7,                    # specify default font size
        # Axes style
        'axes.labelsize' : 9,               # specify axes label size
        'axes.linewidth' : 0.5,             # specify axes line width
        # X-ticks style
        'xtick.direction' : 'in',           # specify x ticks direction
        'xtick.major.size' : 3,             # specify x ticks major size
        'xtick.major.width' : 0.5,          # specify x ticks major width
        'xtick.minor.size' : 1.5,           # specify x ticks minor size
        'xtick.minor.width' : 0.5,          # specify x ticks minor width
        'xtick.minor.visible' : False,      # specify x ticks minor visible
        'xtick.labelsize' : 7,              # specify x ticks label size
        'xtick.top' : True,                 # specify x ticks on top
        # Y-ticks style
        'ytick.direction' : 'in',           # specify y ticks direction
        'ytick.major.size' : 3,             # specify y ticks major size
        'ytick.major.width' : 0.5,          # specify y ticks major width
        'ytick.minor.size' : 1.5,           # specify y ticks minor size
        'ytick.minor.width' : 0.5,          # specify y ticks minor width
        'ytick.minor.visible' : False,      # specify y ticks minor visible
        'ytick.labelsize' : 7,              # specify y ticks label size
        'ytick.right' : True,               # specify y ticks on right
        # Line style
        'lines.linewidth' : 1.0,            # specify line width
        'lines.markersize' : 3,             # specify marker size
        # Grid style
        'grid.linewidth' : 0.5,             # specify grid line width
        # Legend style
        'legend.fontsize' : 6,              # specify legend label size
        'legend.frameon' : False,           # specify legend frame off
        'legend.loc' : 'best',              # specify legend position
        'legend.handlelength' : 2.5,        # specify legend handle length
        # Savefig style
        'savefig.bbox' : 'tight',           # specify savefig bbox
        'savefig.pad_inches' : 0.05,        # specify savefig pad inches
        'savefig.transparent' : True,       # specify savefig transparency
        # Mathtext style
        'mathtext.default' : 'regular',     # specify mathtext font
    }

    # Update matplotlib rcParams
    plt.rcParams.update(MPL_CONFIG)

    # Create the figure
    fig, axs = plt.subplots(
        nrows=n_signs, 
        ncols=n_motifs,
        # sharex=True,
        figsize=(n_motifs*2, n_signs*2)
    )

    # Set y-labels
    for i in range(n_signs):
        axs[i, 0].set_ylabel(r'Covariance $\Sigma_{ij}$')

    # Customize the style of violin plots
    colors = [
        '#8c8cd1', '#7fb2e2', '#7bd4d7', 
        '#7cdd9e', '#92d874', '#b4d645', 
        '#d5ca47', '#f5b43d', '#f58b33'
    ]
    linewidths = [0.5] * 9

    FIG_LABELS = [
        ["(a$+$) $G_{a}$ with positive TIs", "(b$+$) $G_{b}$ with positive TIs", "(c$+$) $G_{c}$ with positive TIs"],
        ["(a$-$) $G_{a}$ with negative TIs", "(b$-$) $G_{b}$ with negative TIs", "(c$-$) $G_{c}$ with negative TIs"]
    ]

    for i in range(n_signs):
        for j in range(n_motifs):
            
            parts = axs[i, j].violinplot(
                COVS[j * n_signs + i], 
                showmedians=False, 
                showmeans=False, 
                showextrema=False
            )
            # Set the color of each violin
            for k, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[k])
                pc.set_edgecolor('black')
                pc.set_linewidths(linewidths[k])

            # Mean
            mean = np.mean(
                COVS[j * n_signs + i],
                axis=0
            )

            # 25th, 50th, and 75th percentiles
            quartile1, medians, quartile3 = np.percentile(
                COVS[j * n_signs + i], 
                [25, 50, 75], 
                axis=0
            )

            # Min and max whiskers
            whiskers_min = np.min(
                COVS[j * n_signs + i], 
                axis=0
            )
            whiskers_max = np.max(
                COVS[j * n_signs + i], 
                axis=0
            )

            # Set the x-axis
            inds = np.arange(1, len(medians) + 1)

            # Plot the mean
            axs[i, j].scatter(
                inds, mean, 
                marker='.', 
                s=4, 
                color='white', 
                zorder=3
            )

            # Plot the percentiles
            axs[i, j].vlines(
                inds, 
                quartile1, quartile3, 
                color='gray', 
                linestyle='-', 
                lw=3
            )

            # Plot the whiskers
            axs[i, j].vlines(
                inds, 
                whiskers_min, 
                whiskers_max, 
                color='gray', 
                linestyle='-', 
                lw=.75
            )

            # Plot the median
            axs[i, j].hlines(
                medians, 
                inds-7.5e-2, inds+7.5e-2, 
                color='black', 
                linestyle='-', 
                lw=.5
            )

            axs[i, j].scatter(
                inds, 
                f_theory[j], 
                marker='x', 
                color='r', 
                label='no TI', 
                zorder=4
            )

            axs[i, j].set_title(
                FIG_LABELS[i][j]
            )
            # Set the legend
            axs[i, j].legend()
            # Set the x-axis 
            axs[i, j].set_xticks(inds)
            axs[i, j].set_xticklabels(LABELS)
        
    # Apply the layout
    fig.tight_layout()

    # Save the figure
    fig.savefig(output_file)

    # Close the figure
    plt.close(fig)

if __name__ == '__main__':
    main()

"""End of file"""