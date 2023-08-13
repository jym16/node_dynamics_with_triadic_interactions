from triadic_interaction import *
import numpy as np
import os 
import matplotlib.pyplot as plt

def main():
    # Define the identifier
    motifs = ["04", "05", "06"]
    n_motifs = len(motifs)
    signs = ["-", "+"]
    n_signs = len(signs)
    identifiers = [m + s for m in motifs for s in signs]
    _bins = 50 # 'fd' if you want to use Freedman-Diaconis rule

    output_file = "./figure/comparison_cond_corr.pdf"

    """Base paths."""
    data_basepaths = ["./data/testcase" + id for id in identifiers]

    """Data filepaths."""
    data_timeseris = [os.path.join(path, "{}_timeseries.npy".format(id)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_123 = [os.path.join(path, "{}_cond_corr_123_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_123_x = [os.path.join(path, "{}_cond_corr_123_x_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_132 = [os.path.join(path, "{}_cond_corr_132_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_132_x = [os.path.join(path, "{}_cond_corr_132_x_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_231 = [os.path.join(path, "{}_cond_corr_231_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_231_x = [os.path.join(path, "{}_cond_corr_231_x_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]

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

    """The simulation parameters."""
    n_folds = 2 # Use 2-fold for stationary distribution
    t0 = -n_timesteps // n_folds # take the second half of the timeseries

    # Theory
    cond_corr_theory_a = [
        lambda x : w_neg / (alpha + w_neg) + 0. * x,
        lambda x : 0. + 0. * x,
        lambda x : 0. + 0. * x
    ]
    cond_corr_theory_b = [
        lambda x : w_neg / np.sqrt((alpha + w_neg) * (alpha + 2 * w_neg)) + 0. * x,
        lambda x : 0. + 0. * x,
        lambda x : w_neg / np.sqrt((alpha + w_neg) * (alpha + 2 * w_neg)) + 0. * x,
    ]
    cond_corr_theory_c = [
        lambda x : w_neg / (alpha + 2 * w_neg) + 0. * x,
        lambda x : w_neg / (alpha + 2 * w_neg) + 0. * x,
        lambda x : w_neg / (alpha + 2 * w_neg) + 0. * x,
    ]
    f_theory = [
        cond_corr_theory_a,
        cond_corr_theory_b,
        cond_corr_theory_c
    ]

    # Load data
    C12_3 = [np.load(data) for data in data_cond_corr_123]
    Xgrid12_3 = [np.load(data) for data in data_cond_corr_123_x]
    C13_2 = [np.load(data) for data in data_cond_corr_132]
    Xgrid13_2 = [np.load(data) for data in data_cond_corr_132_x]
    C23_1 = [np.load(data) for data in data_cond_corr_231]
    Xgrid23_1 = [np.load(data) for data in data_cond_corr_231_x]

    Xs = [np.load(data)[:, t0:, :] for data in data_timeseris]
    RHO = [C23_1, C13_2, C12_3]
    XGRID = [Xgrid23_1, Xgrid13_2, Xgrid12_3]

    # Extract the range of x
    XMIN = []
    XMAX = []
    for i in range(len(identifiers)):
        x_min, x_max = extract_by_std(Xs[i])
        XMIN.append(x_min)
        XMAX.append(x_max)

    MINMAX = [
        (
            min(XMIN[i * n_signs], XMIN[i * n_signs + 1]), 
            max(XMAX[i * n_signs], XMAX[i * n_signs + 1])
        )
        for i in range(n_motifs) 
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
        nrows=n_motifs, 
        ncols=n_nodes, 
        # sharey=True, 
        # sharex=True,
        figsize=(n_nodes*2.5, n_motifs*2)
    )

    COLORS = ['r', 'b']
    LABELS = ['a', 'b']
    MARKERS = ['o', 's']
    LABELS = ['positive', 'negative']

    FIG_LABELS = [(2, 3, 1), (1, 3, 2), (1, 2, 3)]

    for i in range(n_motifs):
        for j in range(n_nodes):
            for k in range(n_signs):
                axs[i, j].scatter(
                    XGRID[j][i * n_signs + k],
                    RHO[j][i * n_signs + k],
                    marker=MARKERS[k],
                    edgecolor=COLORS[k],
                    facecolor='none',
                    alpha=0.6,
                    label=LABELS[k]
                )
            axs[i, j].axhline(
                f_theory[i][j](0), 
                color='k',
                linestyle='--',
                alpha=0.6,
                label='no TIs'
            )

            axs[i, j].set_ylabel(r'$\rho_{}(X_{}, X_{} \mid X_{})$'.format(chr(ord('a') + i), *FIG_LABELS[j]))
            axs[i, j].set_xlabel(r'$X_{}$'.format(j + 1))

            axs[i, j].set_xlim(MINMAX[i])
            axs[i, j].set_ylim([-.5, 1])

            axs[i, j].legend()
        
            # Set the title
            axs[i, j].annotate(
                'motif $G_{}$'.format(chr(ord('a') + i)),
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                fontsize=8, 
                horizontalalignment='left', 
                verticalalignment='top'
            )

    axs[0, 0].axhline(
        w_pos / (w_pos + alpha),
        color='k',
        linestyle=':',
        alpha=0.6
    )
    axs[1, 0].axhline(
        w_pos / np.sqrt((alpha + w_pos) * (alpha + 2 * w_pos)) ,
        color='k',
        linestyle=':',
        alpha=0.6
    )
    axs[2, 0].axhline(
        w_pos / (2 * w_pos + alpha),
        color='k',
        linestyle=':',
        alpha=0.6
    )

    # Set the layout
    fig.tight_layout()

    # Save the figure
    fig.savefig(output_file)

    # Close the figure
    plt.close(fig)

if __name__ == '__main__':
    main()

"""End of file"""