"""
nmpc_baseline.py
Nonlinear MPC baseline on the single-reactor ASM+N2O plant, for a like-for-like
comparison against the convex Koopman-MPC (control performance AND per-step solve
time). Single-shooting: the input trajectory K_La over the horizon is optimized
directly on the true nonlinear model with SLSQP; the objective mirrors the Koopman
controller (aeration energy + fugitive N2O emission + soft ammonia penalty).

This quantifies the price of nonconvexity: the NMPC uses the exact model but is
one to two orders of magnitude slower per step and offers no global-optimality
certificate, whereas the Koopman QP is convex and solves in milliseconds.
"""
import numpy as np, time
from scipy.optimize import minimize
import plant_model as PM
import koopman_mpc as K

DT=K.DT; NH_LIM=K.NH_LIM; KLA_MAX=K.KLA_MAX

class NMPC:
    def __init__(self, Np=12, w_E=1.0, w_N=0.2, rho=8e3, nh_margin=1.0, nsub=2):
        self.Np=Np; self.w_E=w_E; self.w_N=w_N; self.rho=rho
        self.nh_margin=nh_margin; self.nsub=nsub
        self.u_prev=150.0; self.Uwarm=150.0*np.ones(Np)
    def _rollout_cost(self, U, x0, dist):
        x=x0.copy(); J=0.0
        for i in range(self.Np):
            u=U[i]
            J += self.w_E*(u/360.0)
            em,_=PM.emission_rate(x,u); J += self.w_N*em/PM.P['V']   # emission per volume
            viol=max(0.0, x[1]-(NH_LIM-self.nh_margin)); J += self.rho*viol**2
            x=PM.rk4_step(x,u,dist,DT,nsub=self.nsub)
        return J
    def solve(self, x, dist):
        bnds=[(0.0,KLA_MAX)]*self.Np
        t0=time.perf_counter()
        res=minimize(self._rollout_cost, self.Uwarm, args=(x,dist),
                     method='SLSQP', bounds=bnds,
                     options=dict(maxiter=30, ftol=1e-3))
        ms=(time.perf_counter()-t0)*1e3
        U=res.x; self.Uwarm=np.r_[U[1:],U[-1]]        # shift warm start
        u0=float(np.clip(U[0],0,KLA_MAX)); self.u_prev=u0
        return u0, ms

def run_nmpc(x0, nmpc, days=1.5):
    n=int(days/DT); x=x0.copy()
    log=dict(t=[],KLa=[],DO=[],NH=[],TN=[],N2O_em=[],ms=[])
    for k in range(n):
        t=k*DT; NH_in,S_in,Q=PM.influent(t)
        u,ms=nmpc.solve(x,(NH_in,S_in,Q))
        _,=(0,); em,_=PM.emission_rate(x,u)
        log['t'].append(t);log['KLa'].append(u);log['DO'].append(x[0])
        log['NH'].append(x[1]);log['TN'].append(x[1]+x[2]);log['N2O_em'].append(em);log['ms'].append(ms)
        x=PM.rk4_step(x,u,(NH_in,S_in,Q),DT)
    for kk in log: log[kk]=np.array(log[kk])
    return log

if __name__=="__main__":
    x_ss=np.load('x_ss.npy'); km=K.Koopman.load('koop.npz')
    days=1.5
    print("NMPC (nonlinear, single-shooting SLSQP)...")
    nm=NMPC(Np=12,w_E=1.0,w_N=0.2); ln=run_nmpc(x_ss,nm,days=days); mn=K.metrics(ln)
    print("Koopman-MPC (convex QP, emission mode) on same window...")
    mpc=K.KoopMPC(km,w_E=1.0,w_N=0.4,n2o_mode='emission')
    lk,_=K.run_closed_loop(x_ss,mpc,days=days,kind='mpc',forecast='noisy',
                           fc_sigma=0.2,meas_noise=0.02,rng=np.random.default_rng(3))
    mk=K.metrics(lk)
    print("\n%-14s %8s %8s %10s %12s %12s"%("controller","AE","N2O","NHpk","solve_mean","solve_max"))
    for nm_,m in [("NMPC",mn),("Koopman-QP",mk)]:
        print("%-14s %8.0f %8.2f %10.2f %10.1f ms %9.1f ms"%(
            nm_,m['AE_kWh_d'],m['N2O_kgN_d'],m['NH_peak'],m['ms_mean'],m['ms_max']))
    import json; json.dump(dict(NMPC=mn,Koopman=mk,days=days),open('results_nmpc.json','w'),indent=2)
    print("NMPC_DONE")
