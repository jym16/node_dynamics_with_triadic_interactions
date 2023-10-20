from triadic_interaction import *
import numpy as np
import os 
import matplotlib.pyplot as plt

def main():
    # Define the identifier
    identifiers = ["Ra+", "D2a+", "D1a+"]
    _bins = 50 # 'fd' if you want to use Freedman-Diaconis rule

    output_file = "./figure/comparison_gamma-dependence.pdf"

    """Base paths."""
    data_basepaths = ["./data/testcase" + id for id in identifiers]

    """Data filepaths."""
    data_timeseries = [os.path.join(path, "{}_timeseries.npy".format(id)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_231 = [os.path.join(path, "{}_cond_corr_231_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]
    data_cond_corr_231_x = [os.path.join(path, "{}_cond_corr_231_x_{}bins.npy".format(id, _bins)) for path, id in zip(data_basepaths, identifiers)]

    """The structural network."""
    # Number of nodes
    n_cols = 3

    """The model parameters."""
    w_pos = 2.0
    w_neg = 1.0
    threshold = 1.0e-1
    alpha = 1.0e-1
    noise_std = [1.0e-1, 1.0e-2, 1.0e-3]
    dt = 1.0e-2
    t_max = 1.0e2
    n_timesteps = int(1 + t_max / dt)

    """The simulation parameters."""
    n_folds = 2 # Use 2-fold for stationary distribution
    t0 = -n_timesteps // n_folds # take the second half of the timeseries

    # Theory
    f_theory = lambda w, a: w / (a + w)

    # Load the data
    Xs = [np.load(data)[:, t0:, :] for data in data_timeseries]

    C23_1 = [np.load(data) for data in data_cond_corr_231]
    Xgrid23_1 = [np.load(data) for data in data_cond_corr_231_x]

    XMIN = []
    XMAX = []
    for i in range(len(identifiers)):
        x_min, x_max = extract_by_std(Xs[i])
        XMIN.append(x_min)
        XMAX.append(x_max)

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

    n_rows = 1
    n_cols = 3

    # Create the figure
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        # sharey=True,
        figsize=(n_cols*2.25, n_rows*1.75)
    )

    FIG_LABELS = ['(R)', '(D1)', '(D2)']

    positions = [0, 1, 2]
    
    for i in range(len(identifiers)):
        axs[positions[i]].scatter(
            Xgrid23_1[i][(Xgrid23_1[i] >= XMIN[i]) & (Xgrid23_1[i] <= XMAX[i])],
            C23_1[i][(Xgrid23_1[i] >= XMIN[i]) & (Xgrid23_1[i] <= XMAX[i])],
            marker='o',
            edgecolor='r',
            facecolor='none'
        )
        axs[positions[i]].axhline(
            f_theory(w_neg, alpha), 
            color='k',
            linestyle='--',
            label='no TIs'
        )

        axs[positions[i]].set_ylabel(r'$\rho_{}(X_{}, X_{} \mid X_{})$'.format('a', 2, 3, 1))
        axs[positions[i]].set_xlabel(r'$X_{}$'.format(1))
        axs[positions[i]].set_xlim(XMIN[i], XMAX[i])
        axs[positions[i]].set_ylim([.85, 1])

        # Set the title
        axs[positions[i]].annotate(
            FIG_LABELS[i],
            xy=(0.05, 0.95), 
            xycoords='axes fraction',
            fontsize=8, 
            horizontalalignment='left', 
            verticalalignment='top'
        )

    axs[0].axhline(
        w_pos / (w_pos + alpha),
        color='k',
        linestyle=':'
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