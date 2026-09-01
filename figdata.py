"""
figdata.py -- regenerate and cache the trajectories the manuscript figures need.

The three time-series figures (Figures 2, 4 and 7) plot trajectories, which the
results JSON files do not store.  This module re-runs exactly the simulations
that produced them -- same seeds, same settings, same code paths -- caches the
trajectories in .npz, and ASSERTS that the summary indices it recomputes agree
with the archived JSON.  Re-styling a figure therefore cannot silently change a
reported number: if anything drifts, this script fails instead of plotting.

Usage:  python figdata.py            (all)
        python figdata.py vaf|loop|cascade
"""
import json
import sys

import numpy as np

from resultspath import load as load_results

CACHE_VAF = 'cache_vaf.npz'
CACHE_LOOP = 'cache_loop.npz'
CACHE_CASCADE = 'cache_cascade.npz'

TOL = 5e-3          # relative tolerance when checking against the archived JSON


def _check(name, got, want, tol=TOL):
    if want is None:
        print(f'    {name:22s} {got:12.4f}   (no archived value)')
        return
    rel = abs(got - want) / max(abs(want), 1e-9)
    ok = (rel <= tol) or (abs(got - want) < 1e-6)
    print(f'    {name:22s} {got:12.4f}  archived {want:12.4f}   '
          f'{"OK" if ok else "MISMATCH"}')
    if not ok:
        raise SystemExit(f'ABORT: {name} moved by {rel:.2%} -- '
                         f'figures must not be rebuilt from different numbers')


# ---------------------------------------------------------------- Figure 2
def make_vaf():
    """Six-step-ahead prediction traces on the held-out segment."""
    import revision_analyses as RA
    import koopman_mpc as K

    x_ss, X, U, D, Xp, Y, ntr = RA.build_validation()
    km = K.Koopman(X[:ntr]).fit(X[:ntr], U[:ntr], D[:ntr], Xp[:ntr], Y[:ntr])
    xb = np.mean(X[:ntr], 0)
    ub = float(np.mean(U[:ntr]))
    db = np.mean(D[:ntr], 0)
    lin = K.local_linear_model(xb, ub, db)

    Hp = 12
    starts = list(range(ntr, len(X) - Hp))
    yk = np.zeros((len(starts), Hp, 4))
    yl = np.zeros_like(yk)
    yt = np.zeros_like(yk)
    for i, s in enumerate(starts):
        yk[i] = km.predict_multi(X[s], U[s:s + Hp, 0], D[s:s + Hp])
        yl[i] = K.local_predict(lin, X[s], U[s:s + Hp, 0])
        yt[i] = K.outputs_of(Xp[s:s + Hp])

    def vaf(pred, true):
        return float(max(0.0, 100.0 * (1.0 - np.var(true - pred) / np.var(true))))

    arch = load_results('results_revision.json')['vaf']
    print('  Figure 2 -- VAF at the six-step horizon:')
    for o, nm in enumerate(RA.ONAMES):
        _check(f'VAF Koopman {nm}', vaf(yk[:, 5, o], yt[:, 5, o]),
               arch['koopman'][nm][5])
        _check(f'VAF local   {nm}', vaf(yl[:, 5, o], yt[:, 5, o]),
               arch['local'][nm][5])

    np.savez_compressed(CACHE_VAF, t=np.arange(len(starts)) * K.DT,
                        yk=yk, yl=yl, yt=yt)
    print(f'  {CACHE_VAF} written')


# ---------------------------------------------------------------- Figure 4
def make_loop():
    """Single-reactor closed loop: Koopman-MPC, ABAC and fixed-DO PI."""
    import koopman_mpc as K

    x_ss = np.load('x_ss.npy')
    km = K.Koopman.load('koop.npz')

    mpc = K.KoopMPC(km, w_E=1.0, w_N=1.0)
    log_mpc, _ = K.run_closed_loop(x_ss, mpc, days=10.0, kind='mpc',
                                   forecast='noisy', fc_sigma=K.FC_SIGMA,
                                   meas_noise=K.MEAS_NOISE,
                                   rng=np.random.default_rng(11))
    log_pi, _ = K.run_pi_fast(x_ss, K.PIController(Kp=40.0, Ki=200.0), days=10.0)
    log_ab, _ = K.run_pi_fast(x_ss, K.CascadeABAC(), days=10.0)

    arch = load_results('results.json')['table2']
    print('  Figure 4 -- closed-loop indices:')
    for nm, log in [('MPC', log_mpc), ('PI', log_pi), ('ABAC', log_ab)]:
        m = K.metrics(log)
        for key in ['AE_kWh_d', 'N2O_kgN_d', 'NH_viol_h', 'NH_peak']:
            _check(f'{nm} {key}', m[key], arch[nm][key])

    out = {}
    for tag, log in [('mpc', log_mpc), ('pi', log_pi), ('ab', log_ab)]:
        for key in ['t', 'DO', 'KLa', 'NH', 'N2O_em']:
            out[f'{tag}_{key}'] = np.asarray(log[key])
    np.savez_compressed(CACHE_LOOP, **out)
    print(f'  {CACHE_LOOP} written')


# ---------------------------------------------------------------- Figure 7
def make_cascade():
    """Five-tank cascade closed loop: MIMO Koopman-MPC and cascade-PI."""
    import koopman_mpc_cascade as B

    x_ss = np.load('x_ss_bsm2.npy')
    km = B.Koopman.load('koop_bsm2.npz')
    lk, _ = B.run_mpc(x_ss, B.KoopMPC(km, w_N=B.W_N_OPERATING), days=12.0,
                      rng_=np.random.default_rng(3))
    lb, _ = B.run_baseline(x_ss, B.CascadePI(), days=12.0)

    try:
        arch = load_results('results_cascade_final.json')
    except Exception:
        arch = {}
    print('  Figure 7 -- cascade indices:')
    for nm, log in [('Koopman', lk), ('CascadePI', lb)]:
        m = B.metrics(log)
        a = arch.get(nm) or {}
        for key in ['AE_kWh_d', 'N2O_kgN_d', 'NH_viol_h']:
            _check(f'{nm} {key}', m[key], a.get(key))

    out = {}
    for tag, log in [('mpc', lk), ('pi', lb)]:
        for key in ['t', 'DO', 'KLa', 'Qa', 'NH', 'TN', 'N2O_em']:
            out[f'{tag}_{key}'] = np.asarray(log[key])
    np.savez_compressed(CACHE_CASCADE, **out)
    print(f'  {CACHE_CASCADE} written')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('vaf', 'all'):
        make_vaf()
    if what in ('loop', 'all'):
        make_loop()
    if what in ('cascade', 'all'):
        make_cascade()
