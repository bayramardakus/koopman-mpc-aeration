"""
Figures added at revision of ms. 3395422.

  dtfig     Figure 5, control-interval study (reads results_revision.json)
  cascade   Figure 7, closed-loop response of the five-tank cascade,
            MIMO Koopman-MPC against the conventional cascade-PI baseline
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resultspath import load as load_results


def fig_dt():
    rows = load_results('results_revision.json')['dt']
    dt = [r['dt_min'] for r in rows]
    fig, axs = plt.subplots(1, 4, figsize=(14.5, 3.6))

    axs[0].plot(dt, [r['NH_viol'] for r in rows], 'o-', color='#1f77b4')
    axs[0].set_ylabel('NH$_4$ violation time (h / 10 d)')
    axs[0].set_title('Constraint compliance', fontsize=10)

    axs[1].plot(dt, [r['NH_peak'] for r in rows], 'o-', color='#1f77b4')
    axs[1].axhline(4.0, ls='--', color='k', lw=1)
    axs[1].text(6, 4.12, 'permit limit', fontsize=8)
    axs[1].set_ylabel('NH$_4$ peak (mgN L$^{-1}$)')
    axs[1].set_title('Peak effluent ammonia', fontsize=10)

    axs[2].plot(dt, [r['AE'] for r in rows], 'o-', color='#2ca02c', label='AE')
    ax2b = axs[2].twinx()
    ax2b.plot(dt, [r['N2O'] for r in rows], 's--', color='#d62728', label='N$_2$O')
    axs[2].set_ylabel('Aeration energy (kWh d$^{-1}$)', color='#2ca02c')
    ax2b.set_ylabel('N$_2$O (kgN d$^{-1}$)', color='#d62728')
    axs[2].set_title('Operating indices', fontsize=10)

    axs[3].semilogy(dt, [r['ms_mean'] for r in rows], 'o-', color='#9467bd', label='mean')
    axs[3].semilogy(dt, [r['ms_max'] for r in rows], 's--', color='#9467bd', alpha=.6, label='max')
    axs[3].set_ylabel('QP solve time (ms)')
    axs[3].set_title('Computational cost', fontsize=10)
    axs[3].legend(fontsize=8)

    for a in axs:
        a.set_xlabel('control interval (min)')
        a.set_xticks(dt); a.grid(alpha=.3)
    fig.suptitle('Control-interval study: prediction horizon held at three hours of physical time',
                 y=1.03, fontsize=11)
    plt.tight_layout()
    plt.savefig('figR5_interval.png', dpi=140, bbox_inches='tight')
    plt.close()
    print('figR5_interval.png saved')


def fig_cascade():
    import koopman_mpc_cascade as B
    x_ss = np.load('x_ss_cascade.npy')
    km = B.Koopman.load('koop_cascade.npz')
    print('  running MIMO Koopman-MPC ...')
    lk, _ = B.run_mpc(x_ss, B.KoopMPC(km, w_N=B.W_N_OPERATING), days=12.0,
                      rng_=np.random.default_rng(3))
    print('  running cascade-PI baseline ...')
    lb, _ = B.run_baseline(x_ss, B.CascadePI(), days=12.0)

    fig, axs = plt.subplots(6, 1, figsize=(9.5, 13.0), sharex=True)
    spec = [('DO',    'DO (mg O$_2$ L$^{-1}$)',        None),
            ('KLa',   '$K_L a$ (d$^{-1}$)',            None),
            ('Qa',    '$Q_a$ ($\\times$ influent flow)', None),
            ('NH',    'Effluent NH$_4$ (mgN L$^{-1}$)', B.NH_LIM),
            ('TN',    'Effluent TN (mgN L$^{-1}$)',     B.TN_LIM),
            ('N2O_em','N$_2$O emission (gN d$^{-1}$)',  None)]
    for ax, (key, lab, lim) in zip(axs, spec):
        ax.plot(lk['t'], lk[key], color='#1f77b4', lw=1.1, label='MIMO Koopman-MPC')
        ax.plot(lb['t'], lb[key], color='#d62728', lw=1.0, alpha=.85, label='Cascade PI')
        if lim is not None:
            ax.axhline(lim, ls='--', color='k', lw=1)
        ax.set_ylabel(lab, fontsize=9); ax.grid(alpha=.3)
        ax.axvspan(5, 9, color='#4c72b0', alpha=.07)
        ax.axvspan(9, 11, color='#c44e52', alpha=.07)
    axs[0].legend(ncol=2, fontsize=9, loc='upper right')
    axs[-1].set_xlabel('time (d)')
    fig.suptitle('Five-tank ASM1 cascade, 12-day window '
                 '(shaded: rain 5–9 d, storm 9–11 d)', y=0.996, fontsize=11)
    plt.tight_layout()
    plt.savefig('figR7_cascade.png', dpi=140)
    plt.close()
    print('figR7_cascade.png saved')


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('dtfig', 'all'):   fig_dt()
    if st in ('cascade', 'all'): fig_cascade()
