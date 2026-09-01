"""
Graphical abstract for ms. 3395422.

The bar values are read from results_revision.json rather than typed in, so the
abstract cannot drift from the ten-realization statistics reported in Table 11.
Controller colours match Figures 4, 6 and 9.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as S
from resultspath import load as load_results

S.apply()

BLUE, TEAL, DARK, GREY = '#2b6cb0', '#0f766e', '#1a202c', '#5b6a7d'

st = load_results('results_revision.json')['seeds']['stats']
NAMES = ['MPC', 'PI', 'ABAC']
vals = [st[n]['NH_viol_h']['mean'] for n in NAMES]
errs = [st[n]['NH_viol_h']['sd'] for n in NAMES]

fig = plt.figure(figsize=(9.6, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.20)

# ------------------------------------------------------------------ the idea
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.grid(False)


def box(x, y, w, h, t, fc, fs=9.6, ec=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.014,rounding_size=0.03',
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs,
            zorder=3, color=DARK, linespacing=1.5)


def arr(p0, p1, c=DARK, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=14,
                                 lw=lw, color=c, zorder=4,
                                 connectionstyle=f'arc3,rad={rad}'))


box(0.06, 0.66, 0.86, 0.20, 'Activated-sludge plant  (ASM1 + N$_2$O)',
    '#f1f5f9', 10.4)
box(0.06, 0.38, 0.38, 0.18, 'Koopman predictor\nidentified by EDMD', '#dbeafe')
box(0.54, 0.38, 0.38, 0.18, 'strictly convex QP\neach control step', '#dbeafe')
arr((0.25, 0.66), (0.25, 0.56))
arr((0.44, 0.47), (0.54, 0.47))
ax.text(0.30, 0.595, 'data', fontsize=8.6, color=GREY)
box(0.20, 0.10, 0.58, 0.17, 'aeration $K_La$  +  recirculation $Q_a$',
    '#dcfce7', 10.0, ec=TEAL)
arr((0.73, 0.38), (0.73, 0.27), c=TEAL)
# one continuous elbow back to the plant: a plain leg to the corner, then the
# only arrowhead where the path actually arrives. Drawn as two patches with the
# same endpoint, so the corner closes.
ax.add_patch(FancyArrowPatch((0.186, 0.185), (0.045, 0.185), arrowstyle='-',
                             mutation_scale=14, lw=1.8, color=TEAL, zorder=4))
ax.add_patch(FancyArrowPatch((0.045, 0.185), (0.045, 0.66), arrowstyle='-|>',
                             mutation_scale=14, lw=1.8, color=TEAL, zorder=4))
ax.text(0.49, 0.030,
        'ammonia and total-nitrogen limits enter as explicit constraints',
        ha='center', fontsize=8.4, color=GREY)
ax.set_title('A data-driven linear predictor makes each\n'
             'aeration decision a convex program',
             fontsize=11.0, color=DARK, pad=8)

# ---------------------------------------------------------------- the result
ax2 = fig.add_subplot(gs[0, 1])
disp = ['Koopman\nMPC', 'fixed-DO\nPI', 'ammonia\ncascade']
cols = [S.C_MPC, S.C_PI, S.C_ABAC]
ax2.bar(disp, vals, yerr=errs, capsize=5, width=0.62, color=cols, alpha=0.9,
        edgecolor='#334155', lw=0.8,
        error_kw=dict(elinewidth=1.1, ecolor='#222222'))
ax2.set_ylabel('effluent-ammonia violation (h per 10 d)', fontsize=9.4)
ax2.set_title('Ten identification and noise realizations,\n'
              'identical influent and constraints',
              fontsize=11.0, color=DARK, pad=8)
ax2.grid(alpha=0.35, axis='y')
ax2.grid(False, axis='x')
ax2.set_axisbelow(True)
ax2.tick_params(labelsize=9)

# a zero-height bar reads as missing data, so the zero is stated
ax2.plot([-0.31, 0.31], [0, 0], color=S.C_MPC, lw=3.2, solid_capstyle='butt',
         zorder=5, clip_on=False)
ax2.text(0, 0.75, 'none', ha='center', fontsize=11.0, color=S.C_MPC,
         fontweight='bold')
for i in (1, 2):
    ax2.text(i, vals[i] + errs[i] + 0.5, '%.1f h' % vals[i], ha='center',
             fontsize=9.4, color='#334155')
ax2.set_ylim(0, 18.5)

fig.text(0.5, 0.012,
         'The N$_2$O submodel is uncalibrated: emission magnitudes are '
         'model-dependent, the compliance result is not.',
         ha='center', fontsize=8.4, color=GREY, style='italic')
plt.tight_layout(rect=[0, 0.045, 1, 1])
os.makedirs('out', exist_ok=True)
fig.savefig('out/graphical_abstract.png', dpi=400)
fig.savefig('out/graphical_abstract.pdf')
plt.close(fig)
print('graphical_abstract.png / .pdf saved  '
      '(MPC %.2f, PI %.2f, ABAC %.2f h)' % tuple(vals))
