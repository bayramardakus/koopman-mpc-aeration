"""
figstyle.py -- one house style for every figure in ms. 3395422.

Wiley asks for figures supplied as separate high-resolution files.  Everything
here is drawn at journal column widths, with a single type size, a single
colour assignment per controller, and no text that overlaps data.  Each figure
is written twice: a 400-dpi PNG for the manuscript file and a vector PDF for
production.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------- page widths
# Wiley: single column 3.35 in, 1.5 column 4.75 in, double column 6.9 in.
W1, W15, W2 = 3.35, 4.75, 6.9
HMAX = 8.9                      # maximum figure height that fits a journal page

# --------------------------------------------------------------- colours
# One colour per controller, used identically in every figure.
C_MPC = '#1f77b4'               # Koopman-MPC
C_PI = '#d62728'                # fixed-DO PI / cascade-PI
C_ABAC = '#2ca02c'              # ammonia-based cascade
C_PLANT = '#000000'             # nonlinear plant (ground truth)
C_LOCAL = '#d62728'             # local linearization
C_LIMIT = '#333333'             # permit limits
C_RAIN = '#4c72b0'
C_STORM = '#c44e52'
GREY = '#555555'

OUTPUT_COLORS = {'DO': '#0072B2', 'S_NH': '#009E73',
                 'TN': '#CC79A7', 'S_N2O': '#E69F00'}
OUTPUT_LABELS = {'DO': 'DO', 'S_NH': '$S_{\\mathrm{NH}}$',
                 'TN': 'TN', 'S_N2O': '$S_{\\mathrm{N_2O}}$'}


def apply():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 8.5,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9.0,
        'axes.titleweight': 'normal',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#111111',
        'axes.grid': True,
        'grid.color': '#cccccc',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.55,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 8.0,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#bbbbbb',
        'legend.fancybox': False,
        'legend.borderpad': 0.4,
        'lines.linewidth': 1.1,
        'lines.markersize': 4.5,
        'figure.dpi': 110,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,          # embed as TrueType, editable in production
        'ps.fonttype': 42,
        'mathtext.default': 'regular',
    })


def panel(ax, letter, dx=-0.085, dy=1.035, size=9.5):
    """Bold panel letter, placed outside the axes so it never covers data."""
    ax.text(dx, dy, f'({letter})', transform=ax.transAxes,
            fontsize=size, fontweight='bold', va='bottom', ha='left')


def headroom(ax, frac=0.22, bottom=0.04):
    """Grow the upper y-limit so annotations sit in empty space, not on data."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - bottom * span, hi + frac * span)


def annotate_box(ax, x, y, text, ha='right', va='top', color='#111111',
                 size=7.6, weight='normal'):
    """Text with an opaque backing so it is legible wherever it lands."""
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=size, color=color, fontweight=weight, zorder=10,
                   bbox=dict(boxstyle='round,pad=0.28', fc='white',
                             ec='#cccccc', lw=0.6, alpha=0.94))


def weather(ax, rain=(5, 9), storm=(9, 11), label=False):
    """Shade the rain and storm periods identically on every panel."""
    ax.axvspan(*rain, color=C_RAIN, alpha=0.08, lw=0, zorder=0)
    ax.axvspan(*storm, color=C_STORM, alpha=0.08, lw=0, zorder=0)
    if label:
        lo, hi = ax.get_ylim()
        y = hi - 0.06 * (hi - lo)
        ax.text(sum(rain) / 2, y, 'rain', ha='center', va='top',
                fontsize=7.2, color=C_RAIN, style='italic', zorder=6)
        ax.text(sum(storm) / 2, y, 'storm', ha='center', va='top',
                fontsize=7.2, color=C_STORM, style='italic', zorder=6)


def save(fig, stem, outdir='out'):
    """Write the 400-dpi PNG and the vector PDF."""
    import os
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, stem + '.png')
    pdf = os.path.join(outdir, stem + '.pdf')
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    plt.close(fig)
    kb = os.path.getsize(png) / 1024
    print(f'  {stem:26s} png {kb:7.0f} kB   + pdf')
