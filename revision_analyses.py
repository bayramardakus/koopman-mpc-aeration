"""
Analyses added at revision of ms. 3395422 (Water Environment Research).

Stages (run as: python revision_analyses.py <stage>)
  vaf       variance-accounted-for of the Koopman predictor vs local linearization,
            per output and per horizon, plus a time-domain validation figure
  ablation  closed-loop ablation of each component of the controller
  tuning    sensitivity to Np, ridge, n_rbf, r_du (with cross-solver spread)
  seeds     repeated identification/noise seeds -> box-and-whisker statistics
  dt        control-interval study at a fixed 3-hour prediction horizon

Everything reuses the identification and closed-loop code of koopman_mpc.py
unchanged; only the settings named in each stage are varied.
"""
import sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import koopman_mpc as K
import plant_model as PM

OUTFILE = 'results_revision.json'
SC = np.array([3.0, 8.0, 12.0, 0.15])          # output normalisation
ONAMES = ['DO', 'S_NH', 'TN', 'S_N2O']


def _load():
    try:    return json.load(open(OUTFILE))
    except Exception: return {}

def _save(d): json.dump(d, open(OUTFILE, 'w'), indent=2)


# --------------------------------------------------------------------------
class NoLiftKoopman(K.Koopman):
    """Identity 'lift': a linear predictor identified on the raw (normalised)
    state by the same ridge least squares. This is the controller with the
    Koopman lifting removed -- i.e. DMD with control -- and is the ablation
    the reviewer asks for in item 14."""
    def __init__(self, Xtr, ridge=1e-4):
        self.n_rbf = 0
        self.ridge = ridge
        self.centers = np.zeros((0, PM.NX))
        self.sigma = 1.0
        self.N = PM.NX
    def lift(self, X):
        X = np.atleast_2d(X)
        return X / K.SCALE
    def save(self, path): raise NotImplementedError


def build_validation(seed=7, days=14.0, train_frac=0.7):
    """Regenerate exactly the identification/validation split of stage_ident."""
    K.rng = np.random.default_rng(seed)
    x_ss = np.load('x_ss.npy')
    X, U, D, Xp = K.generate_data(x_ss, days=days)
    ntr = int(train_frac * len(X))
    Y = K.outputs_of(X)
    return x_ss, X, U, D, Xp, Y, ntr


# ==========================================================================
def stage_vaf():
    """VAF(%) per output and horizon, Koopman vs local linearisation, on the
    held-out segment; plus a time-domain figure of the 6-step (90-min) forecast."""
    out = _load()
    x_ss, X, U, D, Xp, Y, ntr = build_validation()
    km = K.Koopman(X[:ntr]).fit(X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])
    xb = np.mean(X[:ntr], 0); ub = float(np.mean(U[:ntr])); db = np.mean(D[:ntr], 0)
    lin = K.local_linear_model(xb, ub, db)

    Hp = 12
    starts = list(range(ntr, len(X) - Hp))
    # collect predictions at every horizon for every start
    yk = np.zeros((len(starts), Hp, 4)); yl = np.zeros_like(yk); yt = np.zeros_like(yk)
    for i, s in enumerate(starts):
        yk[i] = km.predict_multi(X[s], U[s:s+Hp, 0], D[s:s+Hp])
        yl[i] = K.local_predict(lin, X[s], U[s:s+Hp, 0])
        yt[i] = K.outputs_of(Xp[s:s+Hp])

    def vaf(pred, true):
        # VAF = 100 (1 - var(e)/var(y)), clipped at 0
        e = true - pred
        return float(max(0.0, 100.0 * (1.0 - np.var(e) / np.var(true))))

    tab = {'horizon_steps': list(range(1, Hp + 1)), 'dt_min': 15,
           'n_validation_windows': len(starts), 'koopman': {}, 'local': {}}
    for o, nm in enumerate(ONAMES):
        tab['koopman'][nm] = [vaf(yk[:, h, o], yt[:, h, o]) for h in range(Hp)]
        tab['local'][nm]   = [vaf(yl[:, h, o], yt[:, h, o]) for h in range(Hp)]
    tab['koopman']['mean'] = [float(np.mean([tab['koopman'][n][h] for n in ONAMES]))
                              for h in range(Hp)]
    tab['local']['mean']   = [float(np.mean([tab['local'][n][h] for n in ONAMES]))
                              for h in range(Hp)]
    out['vaf'] = tab
    for nm in ONAMES + ['mean']:
        print("  VAF %-6s h=1 %5.1f / %5.1f   h=6 %5.1f / %5.1f   h=12 %5.1f / %5.1f  (Koopman/local)"
              % (nm, tab['koopman'][nm][0], tab['local'][nm][0],
                 tab['koopman'][nm][5], tab['local'][nm][5],
                 tab['koopman'][nm][11], tab['local'][nm][11]))

    # ---- time-domain figure: 6-step-ahead forecast traces on the held-out set
    h = 5                                        # index of the 6th step
    t = np.arange(len(starts)) * K.DT
    fig, axs = plt.subplots(4, 1, figsize=(9.5, 9.0), sharex=True)
    lbl = ['DO (mg O$_2$ L$^{-1}$)', 'S$_{NH}$ (mgN L$^{-1}$)',
           'TN (mgN L$^{-1}$)', 'S$_{N_2O}$ (mgN L$^{-1}$)']
    for o in range(4):
        axs[o].plot(t, yt[:, h, o], color='k', lw=1.6, label='nonlinear plant')
        axs[o].plot(t, yk[:, h, o], color='#1f77b4', lw=1.2, label='Koopman predictor')
        axs[o].plot(t, yl[:, h, o], color='#d62728', lw=1.0, ls='--', label='local linearization')
        axs[o].set_ylabel(lbl[o]); axs[o].grid(alpha=.3)
        axs[o].text(0.995, 0.93, 'VAF %.1f%% / %.1f%%' %
                    (tab['koopman'][ONAMES[o]][h], tab['local'][ONAMES[o]][h]),
                    ha='right', va='top', transform=axs[o].transAxes, fontsize=9)
    axs[0].legend(ncol=3, fontsize=9, loc='upper left')
    axs[3].set_xlabel('time on the held-out segment (d)')
    fig.suptitle('Six-step (90-minute) ahead prediction on held-out data', y=0.995)
    plt.tight_layout(); plt.savefig('figR1_vaf_timedomain.png', dpi=140); plt.close()

    # ---- VAF vs horizon figure
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for o, nm in enumerate(ONAMES):
        ax.plot(range(1, Hp+1), tab['koopman'][nm], 'o-', label='Koopman ' + nm)
        ax.plot(range(1, Hp+1), tab['local'][nm], 's--', alpha=.6, label='local ' + nm)
    ax.set_xlabel('prediction step (15 min)'); ax.set_ylabel('VAF (%)')
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig('figR2_vaf_horizon.png', dpi=140); plt.close()
    _save(out); print("STAGE_VAF_DONE")


# ==========================================================================
def _closed_loop(km, days=10.0, forecast="noisy", seed=11, **mpc_kw):
    x_ss = np.load('x_ss.npy')
    mpc = K.KoopMPC(km, **mpc_kw)
    lg, _ = K.run_closed_loop(x_ss, mpc, days=days, kind="mpc", forecast=forecast,
                              fc_sigma=(0.0 if forecast != "noisy" else K.FC_SIGMA),
                              meas_noise=K.MEAS_NOISE,
                              rng=np.random.default_rng(seed))
    return K.metrics(lg)


def stage_ablation():
    """Remove one component at a time from the full controller."""
    out = _load()
    x_ss, X, U, D, Xp, Y, ntr = build_validation()
    km = K.Koopman(X[:ntr]).fit(X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])
    kn = NoLiftKoopman(X[:ntr]).fit(X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])

    rows = []
    def add(name, m, note=""):
        rows.append(dict(variant=name, note=note, AE=m['AE_kWh_d'], N2O=m['N2O_kgN_d'],
                         NH_viol=m['NH_viol_h'], NH_peak=m['NH_peak'],
                         TN_viol=m['TN_viol_h'], EQI=m['EQI_proxy'],
                         ms_mean=m['ms_mean'], ms_max=m['ms_max']))
        print("  %-34s AE=%7.1f N2O=%6.2f viol=%5.2f pk=%5.2f" %
              (name, m['AE_kWh_d'], m['N2O_kgN_d'], m['NH_viol_h'], m['NH_peak']))

    add("full controller", _closed_loop(km), "reference")
    add("no Koopman lift (DMDc predictor)", _closed_loop(kn),
        "RBF observables removed; linear predictor on the raw state")
    add("no influent forecast (persistence)", _closed_loop(km, forecast="persistence"),
        "feed-forward removed")
    add("no offset-free correction", _closed_loop(km, offset_free=False),
        "innovation bias fixed at zero")
    add("dissolved-N2O proxy objective", _closed_loop(km, n2o_mode="dissolved"),
        "emission SLP replaced by the legacy proxy")
    add("myopic horizon (Np=1)", _closed_loop(km, Np=1),
        "prediction removed from the optimisation")
    add("short horizon (Np=4)", _closed_loop(km, Np=4), "")
    out['ablation'] = rows
    _save(out); print("STAGE_ABLATION_DONE")


# ==========================================================================
def stage_tuning():
    """Sensitivity of the closed loop to the design parameters the reviewers
    name: prediction horizon, Tikhonov regularisation, dictionary size and the
    move-suppression weight."""
    out = _load()
    x_ss, X, U, D, Xp, Y, ntr = build_validation()

    # The RBF centres are drawn from the module RNG *after* the excitation data has
    # been generated, exactly as in stage_ident(); restoring that state before every
    # fit keeps the dictionary identical across the sweep, so only the swept
    # parameter changes and the Np = 12 row coincides with Table 2.
    rng_state = K.rng.bit_generator.state
    def fit(n_rbf=40, ridge=1e-4):
        K.rng.bit_generator.state = rng_state
        return K.Koopman(X[:ntr], n_rbf=n_rbf, ridge=ridge).fit(
            X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])

    res = {}
    km0 = fit()

    print(" prediction horizon Np:")
    res['Np'] = []
    for Np in [4, 8, 12, 16, 24]:
        m = _closed_loop(km0, Np=Np)
        res['Np'].append(dict(Np=Np, AE=m['AE_kWh_d'], N2O=m['N2O_kgN_d'],
                              NH_viol=m['NH_viol_h'], NH_peak=m['NH_peak'],
                              ms_mean=m['ms_mean'], ms_max=m['ms_max']))
        print("   Np=%2d AE=%7.1f N2O=%6.2f viol=%5.2f pk=%5.2f solve=%.1f/%.1f ms"
              % (Np, m['AE_kWh_d'], m['N2O_kgN_d'], m['NH_viol_h'], m['NH_peak'],
                 m['ms_mean'], m['ms_max']))
    _save({**out, 'tuning': res})

    print(" Tikhonov ridge:")
    res['ridge'] = []
    for r in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        kmr = fit(ridge=r)
        m = _closed_loop(kmr)
        res['ridge'].append(dict(ridge=r, AE=m['AE_kWh_d'], N2O=m['N2O_kgN_d'],
                                 NH_viol=m['NH_viol_h'], NH_peak=m['NH_peak']))
        print("   ridge=%.0e AE=%7.1f N2O=%6.2f viol=%5.2f pk=%5.2f"
              % (r, m['AE_kWh_d'], m['N2O_kgN_d'], m['NH_viol_h'], m['NH_peak']))
    _save({**out, 'tuning': res})

    print(" dictionary size n_rbf:")
    res['n_rbf'] = []
    for n in [10, 20, 40, 80]:
        kmn = fit(n_rbf=n)
        m = _closed_loop(kmn)
        res['n_rbf'].append(dict(n_rbf=n, N=kmn.N, AE=m['AE_kWh_d'], N2O=m['N2O_kgN_d'],
                                 NH_viol=m['NH_viol_h'], NH_peak=m['NH_peak'],
                                 ms_mean=m['ms_mean'], ms_max=m['ms_max']))
        print("   n_rbf=%3d (N=%3d) AE=%7.1f N2O=%6.2f viol=%5.2f pk=%5.2f solve=%.1f ms"
              % (n, kmn.N, m['AE_kWh_d'], m['N2O_kgN_d'], m['NH_viol_h'], m['NH_peak'],
                 m['ms_mean']))
    out['tuning'] = res
    _save(out); print("STAGE_TUNING_DONE")


def stage_rdu():
    """Move-suppression weight: closed-loop effect and, for each value, the
    spread of the reported indices across OSQP builds (filled in by the caller
    script that runs this under several environments)."""
    out = _load()
    x_ss, X, U, D, Xp, Y, ntr = build_validation()
    km = K.Koopman(X[:ntr]).fit(X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])
    import osqp
    rows = []
    for r in [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
        m = _closed_loop(km, r_du=r)
        rows.append(dict(r_du=r, AE=m['AE_kWh_d'], N2O=m['N2O_kgN_d'],
                         NH_viol=m['NH_viol_h'], NH_peak=m['NH_peak'],
                         EQI=m['EQI_proxy']))
        print("   r_du=%.0e AE=%7.2f N2O=%6.3f viol=%5.2f pk=%6.3f"
              % (r, m['AE_kWh_d'], m['N2O_kgN_d'], m['NH_viol_h'], m['NH_peak']))
    key = 'rdu_osqp_%s_np_%s' % (osqp.__version__, np.__version__)
    out.setdefault('rdu', {})[key] = rows
    _save(out); print("STAGE_RDU_DONE")


# ==========================================================================
def stage_seeds(nseed=10):
    """Repeated identification and noise realisations for all three controllers,
    under a common noise model, so the comparison figure can show distributions."""
    out = _load()
    x_ss = np.load('x_ss.npy')
    rows = {'MPC': [], 'PI': [], 'ABAC': []}
    for sd in range(nseed):
        K.rng = np.random.default_rng(100 + sd)
        X, U, D, Xp = K.generate_data(x_ss, days=14.0)
        ntr = int(0.7 * len(X)); Y = K.outputs_of(X)
        km = K.Koopman(X[:ntr]).fit(X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])
        lg, _ = K.run_closed_loop(x_ss, K.KoopMPC(km), days=10.0, kind="mpc",
                                  forecast="noisy", fc_sigma=K.FC_SIGMA,
                                  meas_noise=K.MEAS_NOISE,
                                  rng=np.random.default_rng(200 + sd))
        m = K.metrics(lg)
        rows['MPC'].append([m['AE_kWh_d'], m['N2O_kgN_d'], m['NH_viol_h'], m['NH_peak']])
        # the baselines see the same measurement noise, so the comparison is like-for-like
        for nm, ctrl in [('PI', K.PIController(Kp=40.0, Ki=200.0)), ('ABAC', K.CascadeABAC())]:
            lb, _ = K.run_pi_fast(x_ss, ctrl, days=10.0, meas_noise=K.MEAS_NOISE,
                                  rng=np.random.default_rng(200 + sd))
            mb = K.metrics(lb)
            rows[nm].append([mb['AE_kWh_d'], mb['N2O_kgN_d'], mb['NH_viol_h'], mb['NH_peak']])
        print("  seed %d done" % sd)

    stats = {}
    labels = ['AE_kWh_d', 'N2O_kgN_d', 'NH_viol_h', 'NH_peak']
    for nm in rows:
        a = np.array(rows[nm])
        stats[nm] = {labels[i]: dict(mean=float(a[:, i].mean()), sd=float(a[:, i].std(ddof=1)),
                                     median=float(np.median(a[:, i])),
                                     q1=float(np.percentile(a[:, i], 25)),
                                     q3=float(np.percentile(a[:, i], 75)),
                                     min=float(a[:, i].min()), max=float(a[:, i].max()),
                                     values=a[:, i].tolist())
                     for i in range(4)}
        print("  %-5s AE %.1f+-%.1f  N2O %.2f+-%.2f  viol %.2f+-%.2f"
              % (nm, stats[nm]['AE_kWh_d']['mean'], stats[nm]['AE_kWh_d']['sd'],
                 stats[nm]['N2O_kgN_d']['mean'], stats[nm]['N2O_kgN_d']['sd'],
                 stats[nm]['NH_viol_h']['mean'], stats[nm]['NH_viol_h']['sd']))
    out['seeds'] = dict(n=nseed, stats=stats)
    _save(out)

    # ---- box-and-whisker + bar figure
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.8))
    names = ['MPC', 'PI', 'ABAC']
    titles = ['Aeration energy (kWh d$^{-1}$)', 'N$_2$O emission (kgN d$^{-1}$)',
              'NH$_4$ violation time (h)', 'NH$_4$ peak (mgN L$^{-1}$)']
    for i, lab in enumerate(labels):
        data = [stats[n][lab]['values'] for n in names]
        bp = axs[i].boxplot(data, tick_labels=names, showmeans=True, widths=.55)
        axs[i].set_title(titles[i], fontsize=10); axs[i].grid(alpha=.3, axis='y')
    fig.suptitle('Distributions over %d identification and noise realisations' % nseed,
                 y=1.02, fontsize=11)
    plt.tight_layout(); plt.savefig('figR3_box.png', dpi=140, bbox_inches='tight'); plt.close()

    fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.6))
    for i, lab in enumerate(labels):
        mu = [stats[n][lab]['mean'] for n in names]
        sd = [stats[n][lab]['sd'] for n in names]
        axs[i].bar(names, mu, yerr=sd, capsize=5,
                   color=['#1f77b4', '#7f7f7f', '#bcbd22'])
        axs[i].set_title(titles[i], fontsize=10); axs[i].grid(alpha=.3, axis='y')
    fig.suptitle('Mean $\\pm$ standard deviation over %d realisations' % nseed,
                 y=1.02, fontsize=11)
    plt.tight_layout(); plt.savefig('figR4_bars.png', dpi=140, bbox_inches='tight'); plt.close()
    print("STAGE_SEEDS_DONE")


# ==========================================================================
def stage_dt():
    """Control-interval study. The prediction horizon is held at three hours in
    physical time, so only the sampling interval changes."""
    out = _load()
    x_ss = np.load('x_ss.npy')
    base_DT = K.DT
    rows = []
    for dt_min in [5.0, 10.0, 15.0, 30.0, 60.0]:
        K.DT = dt_min / 60 / 24
        Np = max(1, int(round(3.0 * 60 / dt_min)))       # 3-hour horizon
        K.rng = np.random.default_rng(7)
        X, U, D, Xp = K.generate_data(x_ss, days=14.0)
        ntr = int(0.7 * len(X)); Y = K.outputs_of(X)
        km = K.Koopman(X[:ntr]).fit(X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])
        # one-step and 3-hour-ahead prediction quality at this interval
        Hp = Np
        err = []
        for s in range(ntr, len(X) - Hp, 5):
            yk = km.predict_multi(X[s], U[s:s+Hp, 0], D[s:s+Hp])
            yt = K.outputs_of(Xp[s:s+Hp])
            err.append(((yk - yt) / SC) ** 2)
        nrmse_3h = float(np.mean(np.sqrt(np.mean(err, 0))[-1]))
        lg, _ = K.run_closed_loop(x_ss, K.KoopMPC(km, Np=Np), days=10.0, kind="mpc",
                                  forecast="noisy", fc_sigma=K.FC_SIGMA,
                                  meas_noise=K.MEAS_NOISE,
                                  rng=np.random.default_rng(11))
        m = K.metrics(lg)
        rows.append(dict(dt_min=dt_min, Np=Np, n_samples=len(X), nrmse_3h=nrmse_3h,
                         AE=m['AE_kWh_d'], N2O=m['N2O_kgN_d'], NH_viol=m['NH_viol_h'],
                         NH_peak=m['NH_peak'], ms_mean=m['ms_mean'], ms_max=m['ms_max'],
                         duty_cycle=float(m['ms_mean'] / (dt_min * 60 * 1000) * 100)))
        print("   dt=%4.0f min Np=%2d NRMSE@3h=%.3f AE=%7.1f N2O=%6.2f viol=%5.2f pk=%5.2f solve=%.1f/%.1f ms"
              % (dt_min, Np, nrmse_3h, m['AE_kWh_d'], m['N2O_kgN_d'], m['NH_viol_h'],
                 m['NH_peak'], m['ms_mean'], m['ms_max']))
    K.DT = base_DT
    out['dt'] = rows
    _save(out); print("STAGE_DT_DONE")


# ==========================================================================
if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("vaf", "all"):      stage_vaf()
    if st in ("ablation", "all"): stage_ablation()
    if st in ("tuning", "all"):   stage_tuning()
    if st in ("rdu", "all"):      stage_rdu()
    if st in ("seeds", "all"):    stage_seeds()
    if st in ("dt", "all"):       stage_dt()
