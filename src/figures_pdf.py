from triadic_interaction import *
import numpy as np
import os 
import matplotlib.pyplot as plt
# from matplotlib import colors
def main():
    # Define the identifier
    motifs = ["04", "05", "06"]
    n_motifs = len(motifs)
    signs = ["-", "+"]
    n_signs = len(signs)
    identifiers = [m + s for m in motifs for s in signs]
    _bins = 50 # 'fd' if you want to use Freedman-Diaconis rule

    logscale = False
    output_file = "./figure/comparison_pdf_log.pdf" if logscale else "./figure/comparison_pdf.pdf"

    # Theory
    def prob_theory_a1(x, Gamma, w, a):
        return np.sqrt(a / (np.pi * Gamma**2)) * np.exp(- a * x**2 / Gamma **2)

    def prob_theory_a23(x, Gamma, w, a):
        return np.sqrt(a * (a + 2. * w) / (np.pi * Gamma**2 * (a + w))) * np.exp(- a * (a + 2. * w) * x**2 / (Gamma **2. * (a + w)))

    def prob_theory_b2(x, Gamma, w, a):
        return np.sqrt(a * (a + 3. * w) / (np.pi * Gamma**2 * (a + w))) * np.exp(- a * (a + 3. * w) * x**2 / (Gamma **2 * (a + w)))

    def prob_theory_b13(x, Gamma, w, a):
        return np.sqrt(a * (a + w) * (a + 3. * w) / (np.pi * Gamma**2 * (a**2 + 3. * a * w + w**2))) * np.exp(- a * (a + w) * (a + 3. * w) * x**2 / (Gamma **2 * (a**2 + 3. * a * w + w**2)))

    def prob_theory_c123(x, Gamma, w, a):
            return np.sqrt(a * (a + 3 * w) / (np.pi * Gamma**2 * (a + w))) * np.exp(- a * (a + 3 * w) * x**2 / (Gamma **2 * (a + w)))

    f_theory = [
        [
            lambda x : prob_theory_a1(x, noise_std, w_neg, alpha),
            lambda x : prob_theory_a23(x, noise_std, w_neg, alpha),
            lambda x : prob_theory_a23(x, noise_std, w_neg, alpha)
        ],
        [
            lambda x : prob_theory_b13(x, noise_std, w_neg, alpha),
            lambda x : prob_theory_b2(x, noise_std, w_neg, alpha),
            lambda x : prob_theory_b13(x, noise_std, w_neg, alpha)
        ],
        [
            lambda x : prob_theory_c123(x, noise_std, w_neg, alpha),
            lambda x : prob_theory_c123(x, noise_std, w_neg, alpha),
            lambda x : prob_theory_c123(x, noise_std, w_neg, alpha)
        ]
    ]

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

    """The simulation parameters."""
    n_folds = 2 # Use 2-fold for stationary distribution
    t0 = -n_timesteps // n_folds # take the second half of the timeseries

    # Load the data and trim them
    Xs = [np.load(data_path)[:, t0:, :] for data_path in data_timeseries]

    # Compute the probability distributions
    PDFS = []
    BINS = []

    for X in Xs:
        pdf_Xs, x_bins = estimate_pdf(X[0].flatten(), bins=_bins)
        pdf_Ys, y_bins = estimate_pdf(X[1].flatten(), bins=_bins)
        pdf_Zs, z_bins = estimate_pdf(X[2].flatten(), bins=_bins)
        _PDFS = [pdf_Xs, pdf_Ys, pdf_Zs]
        _BINS = [x_bins, y_bins, z_bins]
        PDFS.append(_PDFS)
        BINS.append(_BINS)

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
        sharey=True, 
        figsize=(n_nodes*2.25, n_motifs*1.75)
    )

    #  Axis labels
    axs[0, 0].set_ylabel(r'PDF $p^{\rm{st}}_{\rm{a}}(X_{i})$')
    axs[1, 0].set_ylabel(r'PDF $p^{\rm{st}}_{\rm{b}}(X_{i})$')
    axs[2, 0].set_ylabel(r'PDF $p^{\rm{st}}_{\rm{c}}(X_{i})$')
    axs[2, 0].set_xlabel(r'$X_{1}$')
    axs[2, 1].set_xlabel(r'$X_{2}$')
    axs[2, 2].set_xlabel(r'$X_{3}$')

    COLORS = ['r', 'b']
    LABELS = ['a', 'b']
    MARKERS = ['+', 'x']
    LABELS = ['positive', 'negative']

    for i in range(n_motifs):
        for j in range(n_nodes):
            minimum = np.min(np.hstack([BINS[i * 2][j], BINS[i * 2 + 1][j]]))
            maximum = np.max(np.hstack([BINS[i * 2][j], BINS[i * 2 + 1][j]]))
            domain = np.linspace(minimum, maximum, 100)
            axs[i, j].plot(
                domain,
                f_theory[i][j](domain), 
                'k--',
                label='no TI',
                alpha=.6
            )

            for k in range(n_signs):
                axs[i, j].scatter(
                    BINS[i * 2 + k][j],
                    PDFS[i * 2 + k][j],
                    c=COLORS[k],
                    marker=MARKERS[k],
                    label=LABELS[k],
                    alpha=.6
                )

            axs[i, j].legend()

            # Set the yscale to log
            if logscale:
                axs[i, j].set_yscale("log")
        
            # Set the title
            axs[i, j].annotate(
                'motif $G_{}$'.format(chr(ord('a') + i)),
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                fontsize=8, 
                horizontalalignment='left', 
                verticalalignment='top'
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