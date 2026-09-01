"""
Cross-model validation for ms. 3395422 (Reviewer 1, point 5.4).

The controller and its identified Koopman predictor are used EXACTLY as reported in
the paper -- same lift, same weights, same horizon, same solver settings, nothing
re-identified and nothing re-tuned. Only the plant it acts on is replaced, by the
structurally different evaluation model of plant_model_eval.py. The baselines are
run on the same evaluation plant so the comparison is like-for-like.
"""
import numpy as np, json, time
import koopman_mpc as K
import plant_model as PM
import plant_model_eval as EV

DT = K.DT

def _metrics(log):
    dt = float(log['t'][1]-log['t'][0]); days = log['t'][-1]-log['t'][0]
    ae = np.trapezoid(PM.P['So_sat']/(1.8*1000.0)*PM.P['V']*log['KLa'], dx=dt)/days
    n2o = np.trapezoid(log['N2O_em'], dx=dt)/1000.0/days
    NH = np.array(log['NH']); TN = np.array(log['TN'])
    return dict(AE_kWh_d=float(ae), N2O_kgN_d=float(n2o),
                NH_mean=float(NH.mean()), NH_peak=float(NH.max()),
                NH_viol_h=float(np.sum(NH > K.NH_LIM)*dt*24),
                TN_mean=float(TN.mean()),
                TN_viol_h=float(np.sum(TN > K.TN_LIM)*dt*24),
                DO_mean=float(np.mean(log['DO'])),
                ms_mean=float(np.mean([m for m in log['ms'] if m > 0]) if any(m > 0 for m in log['ms']) else 0.0),
                ms_max=float(max(log['ms'])))

def run_eval(x0, controller, days=10.0, kind='mpc', forecast='noisy',
             fc_sigma=0.20, meas_noise=0.02, rng=None, sub_min=0.5):
    """Closed loop on the EVALUATION plant. The controller receives only the seven
    states of the design layout; the hydroxylamine pool is invisible to it."""
    if rng is None: rng = np.random.default_rng(0)
    dt = DT if kind == 'mpc' else 1.0/60/24
    nsub = max(1, int(round(dt*24*60/sub_min)))
    n = int(days/dt); x = x0.copy()
    log = dict(t=[], KLa=[], DO=[], NH=[], TN=[], N2O_em=[], ms=[])
    for k in range(n):
        t = k*dt
        NH_in, S_in, Q = EV.influent(t)
        xd = EV.to_design_state(x)
        xm = xd*(1.0+meas_noise*rng.standard_normal(PM.NX)) if meas_noise > 0 else xd
        if kind == 'mpc':
            Np = controller.Np
            if forecast == 'persistence':
                Dseq = np.tile([NH_in, Q/18446.0], (Np, 1))
            else:
                Dseq = np.array([[EV.influent(t+i*DT)[0], EV.influent(t+i*DT)[2]/18446.0]
                                 for i in range(Np)])
                if forecast == 'noisy' and fc_sigma > 0:
                    err = 1.0+fc_sigma*np.sqrt(np.arange(Np))[:, None]*rng.standard_normal((Np, 2))
                    Dseq = Dseq*np.clip(err, 0.3, 2.0)
            u, ms = controller.solve(xm, Dseq)[:2]
        else:
            u = controller.step(xm); ms = 0.0
        _, em = EV.emission_rate(x, u)
        log['t'].append(t); log['KLa'].append(u); log['DO'].append(x[0])
        log['NH'].append(x[1]); log['TN'].append(x[1]+x[2])
        log['N2O_em'].append(em); log['ms'].append(ms)
        x = EV.rk4_step(x, u, (NH_in, S_in, Q), dt, nsub=nsub)
    for kk in log: log[kk] = np.array(log[kk])
    return log, x

if __name__ == '__main__':
    print('settling the evaluation plant ...', flush=True)
    x_ss = EV.settle(KLa=150.0, days=40.0)
    print('  steady outputs DO=%.2f NH=%.2f TN=%.2f NH2OH=%.4f'
          % (x_ss[0], x_ss[1], x_ss[1]+x_ss[2], x_ss[6]), flush=True)
    km = K.Koopman.load('koop.npz')          # the predictor identified on the DESIGN plant

    out = {}
    print('Koopman-MPC (unchanged) on the evaluation plant ...', flush=True)
    lg, _ = run_eval(x_ss, K.KoopMPC(km, w_E=1.0, w_N=1.0), days=10.0, kind='mpc',
                     forecast='noisy', rng=np.random.default_rng(11))
    out['MPC'] = _metrics(lg)
    print('baselines on the evaluation plant ...', flush=True)
    lp, _ = run_eval(x_ss, K.PIController(Kp=40.0, Ki=200.0), days=10.0, kind='pi',
                     meas_noise=0.0)
    out['PI'] = _metrics(lp)
    la, _ = run_eval(x_ss, K.CascadeABAC(), days=10.0, kind='pi', meas_noise=0.0)
    out['ABAC'] = _metrics(la)

    # influent nitrogen load averaged over the evaluation window, so the emission
    # factor is on the same basis as the reported emission
    tt = np.linspace(0.0, 10.0, 200000)
    Nload = float(np.mean([EV.influent(t_)[0]*EV.influent(t_)[2]/1000.0 for t_ in tt]))
    print('  influent N load over the window: %.1f kgN/d' % Nload, flush=True)
    for nm, m in out.items():
        m['EF_percent'] = 100.0*m['N2O_kgN_d']/Nload
        print('  %-5s AE=%7.1f N2O=%6.2f (EF %.2f%%) NHviol=%5.2f NHpk=%5.2f DO=%.2f'
              % (nm, m['AE_kWh_d'], m['N2O_kgN_d'], m['EF_percent'],
                 m['NH_viol_h'], m['NH_peak'], m['DO_mean']), flush=True)
    json.dump(out, open('results_crossmodel.json', 'w'), indent=2)
    print('CROSSMODEL_DONE')
