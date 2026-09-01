"""
Figure 1 of ms. 3395422, redrawn at revision.

Addresses the reviewer's objection that the previous version appeared to emit two
independent signals from the QP block and did not define K_L a or Q_a: the controller
now returns a single input vector u* = [K_L a, Q_a]^T with both components and their
units labelled, the measured disturbances enter as their own clearly separate path,
and the plant is drawn as the five-tank cascade of which the single reactor is the
one-tank special case.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, GREY, DARK, RED = '#2b6cb0', '#e8edf3', '#1a202c', '#c53030'

def box(ax, x, y, w, h, text, fc=GREY, ec=BLUE, fs=9, lw=1.4, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            zorder=3, color=DARK, weight=weight, linespacing=1.45)

def arrow(ax, p0, p1, text=None, tpos=0.5, dy=0.022, fs=8.2, color=DARK,
          style='-|>', lw=1.5, ls='-', rad=0.0, ha='center'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color, linestyle=ls, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        x = p0[0] + tpos*(p1[0]-p0[0]); y = p0[1] + tpos*(p1[1]-p0[1])
        ax.text(x, y + dy, text, ha=ha, va='bottom', fontsize=fs, color=color, zorder=5)

fig, ax = plt.subplots(figsize=(10.6, 5.5))
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.05, 1.0); ax.axis('off')

# ---------------- offline identification (upper band) ----------------
ax.add_patch(Rectangle((0.035, 0.70), 0.60, 0.245, fc='none', ec='#94a3b8',
                       lw=1.0, ls=(0, (5, 4)), zorder=1))
ax.text(0.045, 0.905, 'Offline, once per re-identification', fontsize=8.5,
        color='#475569', style='italic', zorder=3)

box(ax, 0.06, 0.755, 0.16, 0.105,
    'Excitation data\n$\\{x_j,\\,u_j,\\,d_j,\\,x_j^{+}\\}$', fs=8.6)
box(ax, 0.28, 0.755, 0.155, 0.105,
    'Lift  $\\Psi(x)$\nEq. (12)', fs=8.6)
box(ax, 0.49, 0.755, 0.145, 0.105,
    'EDMD\nEq. (14)–(15)', fs=8.6)
arrow(ax, (0.22, 0.8075), (0.28, 0.8075))
arrow(ax, (0.435, 0.8075), (0.49, 0.8075))

# ---------------- predictor ----------------
box(ax, 0.055, 0.435, 0.30, 0.155,
    'Koopman linear predictor\n'
    '$z^{+}=Az+B_u u+B_d d$,   $\\hat{y}=Cz$\n'
    'Eq. (11)', fc='#dbeafe', fs=9.2, lw=1.7)
arrow(ax, (0.5625, 0.755), (0.36, 0.59), rad=-0.12)
# above the arc, not on it
ax.text(0.505, 0.652, '$A,\\,B_u,\\,B_d,\\,C$', fontsize=8.4, color=DARK,
        ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='none', alpha=0.9))

# ---------------- QP ----------------
box(ax, 0.055, 0.135, 0.30, 0.235,
    'Convex quadratic program\n'
    '$\\min_{U,\\varepsilon}\\ \\frac{1}{2}\\zeta^{\\mathsf{T}}P\\zeta+q^{\\mathsf{T}}\\zeta$\n'
    's.t.  $l \\leq A_c\\zeta \\leq u_c$\n'
    'Eq. (18), (21);   $P \\succ 0$\n'
    'energy + fugitive N$_2$O,\n'
    'NH$_4$ and TN limits (soft, with back-off)',
    fc='#dbeafe', fs=8.8, lw=1.7)
arrow(ax, (0.205, 0.435), (0.205, 0.37))
ax.text(0.215, 0.4025, '$\\Phi,\\,\\Gamma_u,\\,\\Gamma_d$', fontsize=8.4,
        color=DARK, ha='left', va='center')

# ---------------- plant ----------------
box(ax, 0.615, 0.135, 0.345, 0.455, '', fc='white', ec=BLUE, lw=1.7)
ax.text(0.7875, 0.552, 'Activated-sludge plant   $\\dot{x}=f(x,u,d)$',
        ha='center', va='center', fontsize=9.4, color=DARK, weight='bold')
ax.text(0.7875, 0.516, 'ASM1 kinetics + two-pathway N$_2$O,  Eq. (1)–(6)',
        ha='center', va='center', fontsize=8.2, color='#475569')

tank_y, tank_h, tank_w = 0.300, 0.105, 0.051
for i, (lab, fc) in enumerate([('anox\n1', '#f1f5f9'), ('anox\n2', '#f1f5f9'),
                               ('aer\n3', '#bfdbfe'), ('aer\n4', '#bfdbfe'),
                               ('aer\n5', '#bfdbfe')]):
    x = 0.640 + i*(tank_w + 0.006)
    ax.add_patch(Rectangle((x, tank_y), tank_w, tank_h, fc=fc, ec='#64748b', lw=1.0, zorder=3))
    ax.text(x + tank_w/2, tank_y + tank_h/2, lab, ha='center', va='center',
            fontsize=7.4, color=DARK, zorder=4)
ax.add_patch(Rectangle((0.640 + 5*(tank_w+0.006), tank_y), 0.038, tank_h,
                       fc='#f1f5f9', ec='#64748b', lw=1.0, zorder=3))
ax.text(0.640 + 5*(tank_w+0.006) + 0.019, tank_y + tank_h/2, 'settler',
        ha='center', va='center', fontsize=6.6, color=DARK, zorder=4, rotation=90)

# aeration into the three aerated tanks
for i in (2, 3, 4):
    x = 0.640 + i*(tank_w + 0.006) + tank_w/2
    arrow(ax, (x, 0.243), (x, tank_y), lw=1.1, color=BLUE)
ax.text(0.845, 0.214, '$K_L a$ applied to the aerated tanks',
        ha='center', fontsize=7.6, color=BLUE)

# internal recirculation
ax.add_patch(FancyArrowPatch((0.640 + 4*(tank_w+0.006) + tank_w/2, tank_y + tank_h),
                             (0.640 + tank_w/2, tank_y + tank_h),
                             arrowstyle='-|>', mutation_scale=11, lw=1.2, color='#0f766e',
                             connectionstyle="arc3,rad=0.22", zorder=5))
ax.text(0.795, 0.437, '$Q_a$ internal recirculation', fontsize=7.6, color='#0f766e', ha='center')

ax.text(0.7875, 0.168, 'single-reactor reference model = the one-aerated-tank special case',
        ha='center', fontsize=7.6, color='#475569', style='italic')

# ---------------- the single input vector ----------------
arrow(ax, (0.355, 0.2525), (0.615, 0.2525), lw=2.2, color=RED)
ax.text(0.485, 0.293,
        '$u^{\\star}=[\\,K_L a,\\ Q_a\\,]^{\\mathsf{T}}$',
        ha='center', fontsize=10.2, color=RED, weight='bold')
ax.text(0.485, 0.207,
        'one input vector; first block applied\n'
        '$K_L a$: oxygen transfer coefficient (d$^{-1}$)\n'
        '$Q_a$: internal recirculation flow (m$^3$ d$^{-1}$)',
        ha='center', va='top', fontsize=7.7, color=RED)

# ---------------- feedback ----------------
arrow(ax, (0.615, 0.475), (0.355, 0.475), lw=1.6, color=DARK)
ax.text(0.503, 0.492, 'measured outputs $y=[\\,S_O,\\ S_{NH},\\ \\mathrm{TN},\\ S_{N_2O}\\,]^{\\mathsf{T}}$',
        ha='center', fontsize=7.9, color=DARK)

# offset-free correction
box(ax, 0.395, 0.375, 0.175, 0.072,
    'offset-free bias  $\\hat{b}_k$\nEq. (23)', fc='#fef3c7', ec='#b45309', fs=7.8, lw=1.2)
arrow(ax, (0.4825, 0.475), (0.4825, 0.447), lw=1.1, color='#b45309')
arrow(ax, (0.395, 0.411), (0.355, 0.32), lw=1.1, color='#b45309', rad=0.15)

# ---------------- measured disturbance ----------------
box(ax, 0.395, 0.035, 0.175, 0.068,
    'influent forecast\n$d=[\\,S_{NH}^{\\mathrm{in}},\\ Q\\,]^{\\mathsf{T}}$', fc='#dcfce7',
    ec='#15803d', fs=7.8, lw=1.2)
arrow(ax, (0.395, 0.069), (0.355, 0.155), lw=1.3, color='#15803d', rad=-0.1)
arrow(ax, (0.570, 0.069), (0.660, 0.155), lw=1.3, color='#15803d', rad=0.1)
ax.text(0.4825, -0.005, 'measured disturbance — not manipulated',
        ha='center', fontsize=7.4, color='#15803d')

import os
os.makedirs('out', exist_ok=True)
plt.rcParams['pdf.fonttype'] = 42
plt.tight_layout()
plt.savefig('out/Figure1.png', dpi=400, bbox_inches='tight')
plt.savefig('out/Figure1.pdf', bbox_inches='tight')
plt.close()
print('Figure1.png / .pdf saved')
