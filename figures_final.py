"""
figures_final.py -- Figures 2-10 of ms. 3395422, redrawn for the revision.

Nothing here recomputes a result.  Every panel is drawn either from the archived
results JSON or from the trajectory caches written by figdata.py, which asserts
that the trajectories reproduce the archived indices exactly.  Only the drawing
changed: type sizes, colours, panel letters, annotation placement and the
removal of text that sat on top of data.

Usage:  python figures_final.py [2|3|4|5|6|7|8|9|10|all]
"""
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as S
from resultspath import load as load_results

S.apply()


def load_cache(name):
    """Load a trajectory cache, saying how to make it if it is not there."""
    import os
    if not os.path.exists(name):
        raise SystemExit(
            f'{name} is missing. The trajectory caches are build products and are '
            f'not tracked; regenerate them with\n'
            f'    python koopman_mpc.py ident\n'
            f'    python koopman_mpc_cascade.py ident\n'
            f'    python figdata.py\n'
            f'Figures 1, 3, 5, 6, 8, 9 and 10 need no cache and can be drawn now.')
    return np.load(name)

REV = load_results('results_revision.json')
RES = load_results('results.json')
ONAMES = ['DO', 'S_NH', 'TN', 'S_N2O']


# ============================================================== Figure 2
def figure2():
    """Six-step-ahead prediction on the held-out segment."""
    d = load_cache('cache_vaf.npz')
    t, yk, yl, yt = d['t'], d['yk'], d['yl'], d['yt']
    vaf = REV['vaf']
    h = 5                                       # the sixth step

    ylab = ['DO (mg O$_2$ L$^{-1}$)', '$S_{\\mathrm{NH}}$ (mgN L$^{-1}$)',
            'TN (mgN L$^{-1}$)', '$S_{\\mathrm{N_2O}}$ (mgN L$^{-1}$)']

    fig, axs = plt.subplots(4, 1, figsize=(S.W2, 7.4), sharex=True)
    for o, ax in enumerate(axs):
        ax.plot(t, yt[:, h, o], color=S.C_PLANT, lw=1.4, zorder=3,
                label='nonlinear plant')
        ax.plot(t, yk[:, h, o], color=S.C_MPC, lw=1.1, zorder=4,
                label='Koopman predictor')
        ax.plot(t, yl[:, h, o], color=S.C_LOCAL, lw=1.0, ls=(0, (4, 2)),
                zorder=2, label='local linearization')
        ax.set_ylabel(ylab[o])
        ax.margins(x=0.005)
        S.headroom(ax, frac=0.30)
        S.annotate_box(
            ax, 0.988, 0.955,
            'VAF   Koopman %.1f%%\n         local     %.1f%%'
            % (vaf['koopman'][ONAMES[o]][h], vaf['local'][ONAMES[o]][h]))
        S.panel(ax, 'abcd'[o], dx=-0.075, dy=1.01)

    axs[0].legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.10),
                  frameon=False, handlelength=2.4, columnspacing=2.2)
    axs[-1].set_xlabel('time on the held-out segment (d)')
    fig.align_ylabels(axs)
    fig.subplots_adjust(hspace=0.13)
    S.save(fig, 'Figure2')


# ============================================================== Figure 3
def figure3():
    """VAF against prediction horizon, per output."""
    vaf = REV['vaf']
    steps = vaf['horizon_steps']

    fig, ax = plt.subplots(figsize=(S.W15, 3.5))
    for nm in ONAMES:
        c = S.OUTPUT_COLORS[nm]
        ax.plot(steps, vaf['koopman'][nm], '-o', color=c, mfc=c, mec=c,
                ms=4.2, lw=1.4, zorder=4)
        ax.plot(steps, vaf['local'][nm], '--s', color=c, mfc='white', mec=c,
                ms=4.0, lw=1.1, alpha=0.85, zorder=3)

    # two legends: colour = output, line style = predictor
    leg_out = [Line2D([], [], color=S.OUTPUT_COLORS[n], lw=2.2,
                      label=S.OUTPUT_LABELS[n]) for n in ONAMES]
    leg_pred = [Line2D([], [], color=S.GREY, lw=1.4, marker='o', ms=4.2,
                       label='Koopman'),
                Line2D([], [], color=S.GREY, lw=1.1, ls='--', marker='s',
                       ms=4.0, mfc='white', label='local linearization')]
    l1 = ax.legend(handles=leg_out, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   title='output', title_fontsize=8, frameon=False)
    ax.add_artist(l1)
    ax.legend(handles=leg_pred, loc='upper left', bbox_to_anchor=(1.02, 0.46),
              title='predictor', title_fontsize=8, frameon=False)

    ax.set_xlabel('prediction step  (1 step = 15 min)')
    ax.set_ylabel('VAF (%)')
    ax.set_xticks(steps)
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(-3, 105)
    S.save(fig, 'Figure3')


# ============================================================== Figure 4
def figure4():
    """Single-reactor closed loop over the 10-day dry/rain/storm window."""
    d = load_cache('cache_loop.npz')
    m = RES['table2']

    spec = [('DO', 'DO\n(mg O$_2$ L$^{-1}$)', 1.0, 2.0, ':'),
            ('KLa', '$K_L a$\n(d$^{-1}$)', 1.0, None, None),
            ('NH', 'Effluent NH$_4$\n(mgN L$^{-1}$)', 1.0, 4.0, '--'),
            ('N2O_em', 'N$_2$O emission\n(kgN d$^{-1}$)', 1e-3, None, None)]

    fig, axs = plt.subplots(4, 1, figsize=(S.W2, 7.0), sharex=True)
    for i, (key, lab, sc, lim, ls) in enumerate(spec):
        ax = axs[i]
        # the window ends at 10 d, so the storm band is drawn only as far as
        # there are data
        S.weather(ax, rain=(5, 9), storm=(9, 10))
        ax.plot(d['pi_t'], d['pi_' + key] * sc, color=S.C_PI, lw=0.85,
                alpha=0.85, zorder=3, label='fixed-DO PI')
        ax.plot(d['ab_t'], d['ab_' + key] * sc, color=S.C_ABAC, lw=0.95,
                alpha=0.9, zorder=4, label='ABAC cascade')
        ax.plot(d['mpc_t'], d['mpc_' + key] * sc, color=S.C_MPC, lw=1.15,
                zorder=5, label='Koopman-MPC')
        if lim is not None:
            ax.axhline(lim, color=S.C_LIMIT, ls=ls, lw=0.9, zorder=6)
        ax.set_ylabel(lab)
        ax.margins(x=0.004)
        ax.set_xlim(0, 10)
        S.headroom(ax, frac=0.10, bottom=0.03)
        S.panel(ax, 'abcd'[i], dx=-0.105, dy=1.00)

    axs[0].set_ylim(top=6.0)
    axs[0].text(0.15, 5.35, 'dotted line: DO = 2 mg L$^{-1}$, the setpoint of the PI',
                fontsize=7.2, color=S.C_LIMIT, ha='left', va='center')
    axs[2].set_ylim(top=9.0)
    axs[2].text(9.9, 4.15, '4 mgN L$^{-1}$ permit limit', fontsize=7.2,
                color=S.C_LIMIT, ha='right', va='bottom')

    # violation summary where there is empty space, not on the traces
    S.annotate_box(axs[2], 0.988, 0.96,
                   'ammonia-violation time over the window\n'
                   'Koopman-MPC %.1f h    ABAC %.1f h    PI %.1f h'
                   % (m['MPC']['NH_viol_h'], m['ABAC']['NH_viol_h'],
                      m['PI']['NH_viol_h']), size=7.3)

    handles = [Line2D([], [], color=S.C_MPC, lw=1.6, label='Koopman-MPC'),
               Line2D([], [], color=S.C_ABAC, lw=1.6, label='ABAC cascade'),
               Line2D([], [], color=S.C_PI, lw=1.6, label='fixed-DO PI'),
               Patch(fc=S.C_RAIN, alpha=0.22, label='rain (5–9 d)'),
               Patch(fc=S.C_STORM, alpha=0.22, label='storm (from 9 d)')]
    axs[0].legend(handles=handles, ncol=5, loc='lower center',
                  bbox_to_anchor=(0.5, 1.10), frameon=False,
                  handlelength=1.9, columnspacing=1.5)
    axs[-1].set_xlabel('time (d)')
    fig.align_ylabels(axs)
    fig.subplots_adjust(hspace=0.13)
    S.save(fig, 'Figure4')


# ============================================================== Figure 5
def figure5():
    """Control-interval study at a fixed three-hour prediction horizon."""
    rows = REV['dt']
    dt = [r['dt_min'] for r in rows]
    # equal spacing on the abscissa: 5, 10 and 15 min no longer collide
    x = np.arange(len(dt))
    tick = [('%g' % v) for v in dt]

    # 2 x 2 rather than 1 x 4: at journal width a four-panel row leaves each
    # panel too narrow for its axis label and title.
    fig, axs2 = plt.subplots(2, 2, figsize=(S.W2, 4.9))
    axs = axs2.ravel()

    axs[0].plot(x, [r['NH_viol'] for r in rows], 'o-', color=S.C_MPC)
    axs[0].set_ylabel('NH$_4$ violation time (h / 10 d)')
    axs[0].set_title('Constraint compliance')
    S.headroom(axs[0], frac=0.14)

    axs[1].plot(x, [r['NH_peak'] for r in rows], 'o-', color=S.C_MPC)
    axs[1].axhline(4.0, ls='--', color=S.C_LIMIT, lw=0.9)
    axs[1].set_ylabel('NH$_4$ peak (mgN L$^{-1}$)')
    axs[1].set_title('Peak effluent ammonia')
    S.headroom(axs[1], frac=0.16)
    axs[1].text(0.06, 4.15, '4 mgN L$^{-1}$ permit limit', fontsize=7.2,
                color=S.C_LIMIT, va='bottom')

    axs[2].plot(x, [r['AE'] for r in rows], 'o-', color='#2ca02c')
    axs[2].set_ylabel('Aeration energy (kWh d$^{-1}$)', color='#2ca02c')
    axs[2].tick_params(axis='y', colors='#2ca02c')
    ax2b = axs[2].twinx()
    ax2b.plot(x, [r['N2O'] for r in rows], 's--', color=S.C_PI)
    ax2b.set_ylabel('N$_2$O emission (kgN d$^{-1}$)', color=S.C_PI)
    ax2b.tick_params(axis='y', colors=S.C_PI)
    ax2b.grid(False)
    axs[2].set_title('Operating indices')
    # no legend: each series carries the colour of its own axis label, which
    # says which scale it belongs to without a "(left)"/"(right)" gloss
    S.headroom(axs[2], frac=0.14)
    S.headroom(ax2b, frac=0.14)

    axs[3].semilogy(x, [r['ms_mean'] for r in rows], 'o-', color='#9467bd',
                    label='mean')
    axs[3].semilogy(x, [r['ms_max'] for r in rows], 's--', color='#9467bd',
                    alpha=0.55, label='maximum')
    axs[3].set_ylabel('QP solve time (ms)')
    axs[3].set_title('Computational cost')
    axs[3].legend(fontsize=7.2, loc='upper right', handlelength=1.8,
                  borderpad=0.35, labelspacing=0.3)

    for i, a in enumerate(axs):
        a.set_xlabel('control interval (min)')
        a.set_xticks(x)
        a.set_xticklabels(tick)
        a.set_xlim(-0.30, len(dt) - 0.70)
        S.panel(a, 'abcd'[i], dx=-0.16, dy=1.13)

    fig.suptitle('Control-interval study: prediction horizon held at three '
                 'hours of physical time', y=1.01, fontsize=9.0)
    # wider than the default: panel (c) carries a right-hand axis label, which
    # would otherwise sit against panel (d)'s left-hand one
    fig.subplots_adjust(hspace=0.62, wspace=0.52)
    S.save(fig, 'Figure5')


# ============================================================== Figure 6
def figure6():
    """Energy-N2O-compliance frontier traced by the emission weight."""
    par = RES['pareto']
    m = RES['table2']
    AE = np.array([p['AE_kWh_d'] for p in par])
    N2 = np.array([p['N2O_kgN_d'] for p in par])
    wn = [p['w_N'] for p in par]

    fig, ax = plt.subplots(figsize=(S.W15, 3.9))
    ax.plot(AE, N2, '-', color='#9aa5b1', lw=1.2, zorder=2)
    ax.scatter(AE, N2, s=52, color=S.C_MPC, edgecolor='white', linewidth=0.8,
               zorder=4, label='Koopman-MPC frontier (no violation at any $w_N$)')

    # w_N labels alternate above/below so they never sit on the frontier line
    for i, (a, n, w) in enumerate(zip(AE, N2, wn)):
        above = (i % 2 == 0)
        ax.annotate('$w_N$=%g' % w, (a, n), textcoords='offset points',
                    xytext=(0, 10 if above else -19), ha='center',
                    fontsize=7.2, color='#39495c')

    ax.scatter([m['PI']['AE_kWh_d']], [m['PI']['N2O_kgN_d']], marker='*',
               s=250, color=S.C_PI, edgecolor='white', linewidth=0.8, zorder=5,
               label='fixed-DO PI (%.1f h violation)' % m['PI']['NH_viol_h'])
    ax.scatter([m['ABAC']['AE_kWh_d']], [m['ABAC']['N2O_kgN_d']], marker='P',
               s=140, color=S.C_ABAC, edgecolor='white', linewidth=0.8, zorder=5,
               label='ABAC cascade (%.1f h violation)' % m['ABAC']['NH_viol_h'])

    ax.annotate('PI', (m['PI']['AE_kWh_d'], m['PI']['N2O_kgN_d']),
                textcoords='offset points', xytext=(12, 2), fontsize=8,
                color=S.C_PI, fontweight='bold', va='center')
    ax.annotate('ABAC', (m['ABAC']['AE_kWh_d'], m['ABAC']['N2O_kgN_d']),
                textcoords='offset points', xytext=(12, 0), fontsize=8,
                color=S.C_ABAC, fontweight='bold', va='center')

    ax.set_xlabel('Aeration energy (kWh d$^{-1}$)')
    ax.set_ylabel('N$_2$O emission (kgN d$^{-1}$)')
    ax.margins(x=0.10, y=0.13)
    # the legend goes below the axes: inside, it covers either the PI marker or
    # the frontier itself, whichever corner it is placed in
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1,
              fontsize=7.6, handletextpad=0.6, borderpad=0.5,
              labelspacing=0.45, frameon=False)
    S.save(fig, 'Figure6')


# ============================================================== Figure 7
def figure7():
    """Five-tank cascade closed loop over the twelve-day window."""
    d = load_cache('cache_cascade.npz')
    fin = load_results('results_cascade_final.json')

    spec = [('DO', 'DO\n(mg O$_2$ L$^{-1}$)', 1.0, None),
            ('KLa', '$K_L a$\n(d$^{-1}$)', 1.0, None),
            ('Qa', '$Q_a/Q_{\\mathrm{in}}$', 1.0, None),
            ('NH', 'Effluent NH$_4$\n(mgN L$^{-1}$)', 1.0, 4.0),
            ('TN', 'Effluent TN\n(mgN L$^{-1}$)', 1.0, 18.0),
            ('N2O_em', 'N$_2$O emission\n(kgN d$^{-1}$)', 1e-3, None)]

    # the recirculation panel needs less height: both controllers hold it
    # constant, so the panel carries one number each
    fig, axs = plt.subplots(6, 1, figsize=(S.W2, 8.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 0.55, 1, 1, 1]))
    for i, (key, lab, sc, lim) in enumerate(spec):
        ax = axs[i]
        S.weather(ax)
        ax.plot(d['pi_t'], d['pi_' + key] * sc, color=S.C_PI, lw=0.95,
                alpha=0.9, zorder=3, label='cascade-PI')
        ax.plot(d['mpc_t'], d['mpc_' + key] * sc, color=S.C_MPC, lw=1.05,
                zorder=4, label='MIMO Koopman-MPC')
        if lim is not None:
            ax.axhline(lim, color=S.C_LIMIT, ls='--', lw=0.9, zorder=5)
        ax.set_ylabel(lab)
        ax.margins(x=0.004)
        S.panel(ax, 'abcdef'[i], dx=-0.105, dy=0.99)

    axs[2].set_ylim(0.35, 2.20)                     # was 80% empty space
    axs[3].set_ylim(top=8.0)
    axs[3].text(11.9, 4.15, '4 mgN L$^{-1}$ limit', fontsize=7.0,
                color=S.C_LIMIT, ha='right', va='bottom')
    axs[4].set_ylim(top=21.5)
    axs[4].text(11.9, 18.3, '18 mgN L$^{-1}$ limit', fontsize=7.0,
                color=S.C_LIMIT, ha='right', va='bottom')

    S.annotate_box(axs[3], 0.988, 0.97,
                   'ammonia-violation time   Koopman-MPC %.1f h    '
                   'cascade-PI %.1f h'
                   % (fin['Koopman']['NH_viol_h'], fin['CascadePI']['NH_viol_h']),
                   size=7.2)

    handles = [Line2D([], [], color=S.C_MPC, lw=1.6, label='MIMO Koopman-MPC'),
               Line2D([], [], color=S.C_PI, lw=1.6, label='cascade-PI'),
               Patch(fc=S.C_RAIN, alpha=0.22, label='rain (5–9 d)'),
               Patch(fc=S.C_STORM, alpha=0.22, label='storm (9–11 d)')]
    axs[0].legend(handles=handles, ncol=4, loc='lower center',
                  bbox_to_anchor=(0.5, 1.08), frameon=False,
                  handlelength=1.9, columnspacing=1.6)
    axs[-1].set_xlabel('time (d)')
    fig.align_ylabels(axs)
    fig.subplots_adjust(hspace=0.12)
    S.save(fig, 'Figure7')


# ============================================================== Figure 8
def figure8():
    """Cascade frontiers: each controller swept across its own aggressiveness."""
    fr = load_results('results_cascade_frontier.json')
    k, c = fr['koopman'], fr['cascade']
    kAE = [m['AE_kWh_d'] for m in k]
    cAE = [m['AE_kWh_d'] for m in c]
    # sweep settings, in the order eval_cascade.frontier() runs them
    k_bo = [0.0, 0.7, 1.5, 2.2]                               # ammonia back-off
    c_set = [(3.5, 2.5), (2.0, 3.2), (1.0, 4.0), (0.4, 4.5)]  # NH4 target / DO ceiling

    fig, axs = plt.subplots(1, 2, figsize=(S.W2, 3.2))
    for i, (key, ylab, title) in enumerate(
            [('NH_viol_h', 'NH$_4$ violation time (h / 5 d)',
              'Energy against ammonia compliance'),
             ('N2O_kgN_d', 'N$_2$O emission (kgN d$^{-1}$)',
              'Energy against N$_2$O')]):
        ax = axs[i]
        ax.plot(cAE, [m[key] for m in c], 's--', color=S.C_PI, ms=5,
                label='cascade-PI (NH$_4$ target / DO ceiling swept)')
        ax.plot(kAE, [m[key] for m in k], 'o-', color=S.C_MPC, ms=5,
                label='Koopman-MPC (ammonia back-off swept)')
        ax.set_xlabel('Aeration energy (kWh d$^{-1}$)')
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.margins(x=0.14, y=0.20)
        S.panel(ax, 'ab'[i], dx=-0.20, dy=1.08)

    # label the two ends of the cascade-PI sweep, and the MPC cluster once
    axs[0].annotate('NH$_4$ target %.1f / DO ceiling %.1f' % c_set[0],
                    (cAE[0], c[0]['NH_viol_h']), textcoords='offset points',
                    xytext=(7, -1), ha='left', va='center', fontsize=6.8,
                    color=S.C_PI)
    axs[0].annotate('%.1f / %.1f' % c_set[-1],
                    (cAE[-1], c[-1]['NH_viol_h']), textcoords='offset points',
                    xytext=(7, -3), ha='left', fontsize=6.8, color=S.C_PI)
    # the MPC cluster sits on the zero line, where a label would collide with
    # the tick labels; it goes in the empty upper right of the panel instead,
    # with a leader down to the cluster it describes
    axs[0].annotate('back-off %.1f\u2013%.1f mgN L$^{-1}$:\nno violation at any setting'
                    % (k_bo[0], k_bo[-1]),
                    (float(np.mean(kAE)), 0.0),
                    xycoords='data', textcoords='axes fraction',
                    xytext=(0.62, 0.52), ha='center', va='center',
                    fontsize=6.8, color=S.C_MPC,
                    arrowprops=dict(arrowstyle='-', color=S.C_MPC, lw=0.7,
                                    shrinkB=6))

    # one legend for both panels, below the axes, so neither plot carries a box
    # over its data
    handles, labs = axs[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=2, fontsize=7.2,
               handlelength=1.8, columnspacing=2.2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.34, bottom=0.26)
    S.save(fig, 'Figure8')


# ============================================================== Figure 9
def figure9():
    """Ten-realization statistics: distributions above, mean +/- SD below."""
    st = REV['seeds']['stats']
    n = REV['seeds']['n']
    names = ['MPC', 'PI', 'ABAC']
    disp = ['MPC', 'PI', 'ABAC']
    cols = [S.C_MPC, S.C_PI, S.C_ABAC]
    labels = ['AE_kWh_d', 'N2O_kgN_d', 'NH_viol_h', 'NH_peak']
    titles = ['Aeration energy\n(kWh d$^{-1}$)', 'N$_2$O emission\n(kgN d$^{-1}$)',
              'NH$_4$ violation time\n(h)', 'NH$_4$ peak\n(mgN L$^{-1}$)']

    fig, axs = plt.subplots(2, 4, figsize=(S.W2, 4.9))
    for j, lab in enumerate(labels):
        data = [st[nm][lab]['values'] for nm in names]

        ax = axs[0, j]
        bp = ax.boxplot(data, tick_labels=disp, showmeans=True, widths=0.55,
                        patch_artist=True,
                        medianprops=dict(color='#222222', lw=1.2),
                        meanprops=dict(marker='D', mfc='white', mec='#222222',
                                       ms=4.0),
                        flierprops=dict(marker='o', ms=3, mfc='none',
                                        mec='#666666'))
        for box, c in zip(bp['boxes'], cols):
            box.set(facecolor=c, alpha=0.35, edgecolor=c, lw=1.1)
        ax.set_title(titles[j], fontsize=8.2)
        ax.grid(axis='x', visible=False)
        S.panel(ax, 'abcd'[j], dx=-0.34, dy=1.16)

        ax = axs[1, j]
        mu = [st[nm][lab]['mean'] for nm in names]
        sd = [st[nm][lab]['sd'] for nm in names]
        ax.bar(disp, mu, yerr=sd, capsize=4, color=cols, alpha=0.85,
               edgecolor='white', linewidth=0.6,
               error_kw=dict(elinewidth=1.0, ecolor='#222222'))
        ax.grid(axis='x', visible=False)
        S.panel(ax, 'efgh'[j], dx=-0.34, dy=1.04)

    axs[0, 0].set_ylabel('distribution over %d realizations' % n, fontsize=7.8)
    axs[1, 0].set_ylabel('mean $\\pm$ standard deviation', fontsize=7.8)
    for ax in axs.ravel():
        ax.tick_params(axis='x', labelsize=7.8)

    fig.legend(handles=[Line2D([], [], color='#222222', lw=1.2, label='median'),
                        Line2D([], [], color='none', marker='D', mfc='white',
                               mec='#222222', ms=4.0, label='mean')],
               loc='lower center', bbox_to_anchor=(0.5, 0.985), ncol=2,
               frameon=False, fontsize=7.6)
    fig.subplots_adjust(hspace=0.55, wspace=0.42)
    S.save(fig, 'Figure9')


# ============================================================== Figure 10
def figure10():
    """Steady-state N2O characteristic of the design and evaluation plants."""
    rows = load_results('results_crossmodel_char.json')
    do_d = [r['DO_design'] for r in rows]
    ef_d = [r['EF_design'] for r in rows]
    do_e = [r['DO_eval'] for r in rows]
    ef_e = [r['EF_eval'] for r in rows]

    fig, ax = plt.subplots(figsize=(S.W15, 3.5))
    ax.axhspan(1.0, 1.6, color='#8896a8', alpha=0.16, lw=0, zorder=0)
    ax.axvspan(1.09, 1.81, color=S.C_ABAC, alpha=0.13, lw=0, zorder=0)

    ax.plot(do_d, ef_d, 'o-', color=S.C_MPC, ms=4.5, lw=1.4, zorder=4,
            label='design model: two-pathway AOB,\ninhibition-ratio DO law, no N$_2$O sink')
    ax.plot(do_e, ef_e, 's--', color=S.C_PI, ms=4.5, lw=1.3, zorder=4,
            label='evaluation model: explicit NH$_2$OH,\nHaldane DO law, N$_2$O reductase')

    pd_ = do_d[int(np.argmax(ef_d))]
    pe_ = do_e[int(np.argmax(ef_e))]
    ax.annotate('design model:\nmaximum at DO $\\approx$ %.1f' % pd_,
                (pd_, max(ef_d)), textcoords='offset points', xytext=(26, 6),
                fontsize=7.4, color=S.C_MPC, va='center',
                arrowprops=dict(arrowstyle='-', color=S.C_MPC, lw=0.8))
    # below the evaluation curve rather than above it: placed above, the leader
    # had to cross the design curve to reach its own peak
    ax.annotate('evaluation model:\nmaximum at DO $\\approx$ %.1f' % pe_,
                (pe_, max(ef_e)), textcoords='offset points', xytext=(34, -46),
                fontsize=7.4, color=S.C_PI, va='center', ha='left',
                arrowprops=dict(arrowstyle='-', color=S.C_PI, lw=0.8,
                                shrinkB=4))

    ax.set_xlabel('mean dissolved oxygen (mg O$_2$ L$^{-1}$)')
    ax.set_ylabel('emission factor (% of influent N)')
    ax.set_xlim(-0.1, 5.6)
    ax.margins(y=0.10)

    handles, labs = ax.get_legend_handles_labels()
    handles += [Patch(fc='#8896a8', alpha=0.3,
                      label='reported full-scale range 1.0–1.6%'),
                Patch(fc=S.C_ABAC, alpha=0.25,
                      label='closed-loop mean DO on the\nevaluation plant (1.09–1.81)')]
    labs += [h.get_label() for h in handles[-2:]]
    # below the axes: inside, a four-entry legend covers the tail of both curves
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.20),
              ncol=2, fontsize=7.0, labelspacing=0.5, handlelength=1.8,
              columnspacing=1.6, frameon=False)
    S.save(fig, 'Figure10')


FIGS = {'2': figure2, '3': figure3, '4': figure4, '5': figure5, '6': figure6,
        '7': figure7, '8': figure8, '9': figure9, '10': figure10}

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    keys = FIGS.keys() if what == 'all' else [what]
    for k in keys:
        FIGS[k]()
