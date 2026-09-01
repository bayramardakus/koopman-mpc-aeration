"""
koopman_mpc_cascade.py  (v2)
MIMO Koopman-operator MPC for the ASM1-calibrated 5-tank BSM2-style cascade
(plant_model_cascade.py). Preserves the convex-QP / EDMD-ridge / offset-free core.

Features
  * MIMO input u = [K_La, Q_a]  (aeration + internal recirculation).
  * N2O objective is the ACTUAL fugitive emission alpha*K_La*S_N2O (bilinear in the
    input and a lifted output), handled by successive linearization (SLP, 2 iters
    per step). Each SLP pass is a convex QP; convexity and warm-starting are kept.
  * DO-setpoint tracking on the last aerated tank pins the realistic operating point.
  * Explicit soft NH4/TN constraints with a (dynamic) compliance back-off.
  * Offset-free output-disturbance correction; 15-min NH4/N2O measurement delay
    handled in the closed-loop runner.
  * Conventional MIMO baseline: cascade DO-PI with an ammonia-based DO-setpoint
    outer loop and a nitrate/TN-based internal-recirculation loop.
"""
import numpy as np, time, json
if not hasattr(np,'trapezoid'): np.trapezoid=np.trapz  # numpy 1.x/2.x compat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import sparse
import osqp
from koopman_mpc import osqp_settings

R_DU_BSM2 = (3e-2, 3e-2)   # move-suppression weights (set by the tuning study)

# Plant integration sub-step, common to every runner so that the controller and the
# baselines are compared on the same numerical experiment. 0.5 min; the reported
# indices are converged at 1 min and unchanged down to 0.125 min.
# Emission weight at the reported operating point of the cascade study. On this plant
# any w_N > 0 drives the aeration input onto its upper bound for a marginal emission
# gain, and w_N >= 2 destabilises the loop; see the sweep reported in the paper.
W_N_OPERATING = 0.0

SUBSTEP_MIN = 0.5
def SUBSTEPS(dt_days):
    return max(1, int(round(dt_days*24*60/SUBSTEP_MIN)))
import plant_model_cascade as PM

rng = np.random.default_rng(7)
DT = 15.0/60/24
NU = 2
KLA_MAX = 260.0
QA_MAX = PM.Q_A_MAX
NH_LIM, TN_LIM = 4.0, 18.0
DO_REF = 2.0
T = PM.NCOMP                                   # 8 components/tank
SCALE = np.tile(np.array([3.0,40.0,10.0,12.0,120.0,2500.0,180.0,0.12]), PM.NTANK)
VAER_TOT = float(np.sum(PM.VOL[PM.AER]))
K_EM = PM.P['alpha_n2o']*VAER_TOT              # emission = K_EM * KLa * y_N2O  (gN/d)

def outputs_of(X):
    """y = [DO(tank5), NH_eff, N_tot_eff, N2O_dissolved(vol-weighted aerated)]."""
    X=np.atleast_2d(X)
    DO=X[:,4*T+PM.CO['S_O']]; NH=X[:,4*T+PM.CO['S_NH']]; NO=X[:,4*T+PM.CO['S_NO']]
    n2o=sum(PM.VOL[i]*X[:,i*T+PM.CO['S_N2O']] for i in PM.AER)/VAER_TOT
    return np.stack([DO,NH,NH+NO,n2o],axis=1)

def settle(u=(120.0,2.0*PM.P['Q0']), days=30.0):
    x=PM.initial_state()
    for _ in range(int(days/DT)): x=PM.rk4_step(x,list(u),PM.influent(3.0),DT,nsub=4)
    return x

# ------------------------- excitation data (PRBS on both inputs) -------------------------
def generate_data(x0, days=12.0):
    n=int(days/DT); X=[];U=[];D=[];Xp=[]; x=x0.copy(); KLa=120.0; Qa=2.0*PM.P['Q0']; hold=0
    for k in range(n):
        t=k*DT
        if hold<=0:
            KLa=float(np.clip(rng.uniform(60,240)+rng.normal(0,15),40,KLA_MAX))
            Qa =float(np.clip(rng.uniform(0.5,4.0)*PM.P['Q0'],0,QA_MAX)); hold=rng.integers(2,8)
        hold-=1
        S_S,NH_in,X_S,X_BH,Q=PM.influent(t); d=np.array([NH_in,Q/PM.P['Q0']])
        X.append(x.copy()); U.append([KLa,Qa]); D.append(d)
        x=PM.rk4_step(x,[KLa,Qa],(S_S,NH_in,X_S,X_BH,Q),DT,nsub=4); Xp.append(x.copy())
    return map(np.array,(X,U,D,Xp))

# ------------------------- EDMD (ridge) -------------------------
class Koopman:
    def __init__(self, Xtr, n_rbf=40, ridge=1e-2):   # heavier ridge stabilizes stiff DO mode
        self.n_rbf=n_rbf; self.ridge=ridge; Xn=Xtr/SCALE
        idx=rng.choice(len(Xn),size=n_rbf,replace=False); self.centers=Xn[idx]
        dd=np.linalg.norm(self.centers[:,None,:]-self.centers[None,:,:],axis=2)
        self.sigma=np.median(dd[dd>0]); self.N=PM.NX+n_rbf
    def lift(self,X):
        X=np.atleast_2d(X); Xn=X/SCALE
        d2=np.sum((Xn[:,None,:]-self.centers[None,:,:])**2,axis=2)
        return np.hstack([Xn, np.exp(-d2/(2*self.sigma**2))])
    def fit(self,X,U,D,Xp,Y):
        Z=self.lift(X); Zp=self.lift(Xp); Om=np.hstack([Z,U,D]); n=Z.shape[1]; m=Om.shape[1]
        AB=(np.linalg.solve(Om.T@Om+self.ridge*np.eye(m), Om.T@Zp)).T
        self.A=AB[:,:n]; self.Bu=AB[:,n:n+NU]; self.Bd=AB[:,n+NU:]
        self.C=(np.linalg.solve(Z.T@Z+self.ridge*np.eye(n), Z.T@Y)).T; return self
    def predict_multi(self,x0,Useq,Dseq):
        z=self.lift(x0).ravel(); ys=[]
        for u,d in zip(Useq,Dseq): z=self.A@z+self.Bu@u+self.Bd@d; ys.append(self.C@z)
        return np.array(ys)
    def save(self,p): np.savez(p,A=self.A,Bu=self.Bu,Bd=self.Bd,C=self.C,
                               centers=self.centers,sigma=self.sigma,ridge=self.ridge,n_rbf=self.n_rbf)
    @classmethod
    def load(cls,p):
        d=np.load(p); o=cls.__new__(cls)
        o.A=d['A'];o.Bu=d['Bu'];o.Bd=d['Bd'];o.C=d['C'];o.centers=d['centers']
        o.sigma=float(d['sigma']);o.ridge=float(d['ridge']);o.n_rbf=int(d['n_rbf']);o.N=o.A.shape[0]; return o

# ------------------------- MIMO condensed-QP MPC with emission-SLP -------------------------
class KoopMPC:
    def __init__(self, km, Np=12, w_air=1.0, w_pump=0.3, w_N=1.0, q_do=6.0, do_ref=DO_REF,
                 r_du=None, rho=8e3, eps_s=1.0, backoff_nh=1.0, backoff_tn=1.0,
                 dynamic_backoff=True, slp_iters=2):
        self.km=km; self.Np=Np; self.m=NU; self.do_ref=do_ref
        self.w_air=w_air; self.w_pump=w_pump; self.w_N=w_N; self.q_do=q_do
        self.r_du=np.array(R_DU_BSM2 if r_du is None else r_du); self.rho=rho; self.eps_s=eps_s
        self.backoff_nh0=backoff_nh; self.backoff_tn=backoff_tn
        self.dynamic_backoff=dynamic_backoff; self.slp_iters=slp_iters
        self.u_prev=np.array([120.0,2.0*PM.P['Q0']]); self.bias=np.zeros(4)
        self.y_pred_next=None; self.innov_nh_var=0.0; self.warm=None
        self._build_static(); self._prob=None
    def _build_static(self):
        A,Bu,C=self.km.A,self.km.Bu,self.km.C; N=A.shape[0]; Np=self.Np; m=self.m; p=4
        Ap=[np.linalg.matrix_power(A,i) for i in range(Np+1)]; CA=[C@Ap[i] for i in range(Np+1)]
        Phi=np.zeros((Np*p,N)); Gu=np.zeros((Np*p,Np*m))
        for i in range(Np):
            Phi[i*p:(i+1)*p,:]=CA[i+1]
            for j in range(i+1): Gu[i*p:(i+1)*p, j*m:(j+1)*m]=CA[i-j]@Bu
        self.Phi=Phi; self.Gu=Gu
        r=lambda o:[i*p+o for i in range(Np)]
        self.iDO,self.iNH,self.iTN,self.iN2O=r(0),r(1),r(2),r(3)
        self.Gu_do=Gu[self.iDO]; self.Gu_nh=Gu[self.iNH]; self.Gu_tn=Gu[self.iTN]; self.Gu_n2o=Gu[self.iN2O]
        self.Phi_do=Phi[self.iDO]; self.Phi_nh=Phi[self.iNH]; self.Phi_tn=Phi[self.iTN]; self.Phi_n2o=Phi[self.iN2O]
        Dd=np.zeros((Np*m,Np*m))
        for j in range(Np):
            for c in range(m):
                rr=j*m+c; Dd[rr,rr]=1.0
                if j>0: Dd[rr,(j-1)*m+c]=-1.0
        self.Dd=Dd
        self.sel_air=np.array([1.0,0.0]*Np); self.sel_pump=np.array([0.0,1.0]*Np)
        self.kla_pos=np.arange(0,Np*m,m)              # indices of K_La entries in U
        Rdu=Dd.T@np.diag(np.tile(self.r_du,Np))@Dd
        P_UU=2*(self.q_do*self.Gu_do.T@self.Gu_do + Rdu)    # emission term is LINEAR (SLP)
        big=np.block([[P_UU,np.zeros((Np*m,2*Np))],[np.zeros((2*Np,Np*m)),2*self.eps_s*np.eye(2*Np)]])
        self.P=sparse.csc_matrix((big+big.T)/2)
        I_u=np.eye(Np*m); I_n=np.eye(Np)
        A_box=np.hstack([I_u,np.zeros((Np*m,2*Np))])
        A_nh=np.hstack([self.Gu_nh,-I_n,np.zeros((Np,Np))]); A_tn=np.hstack([self.Gu_tn,np.zeros((Np,Np)),-I_n])
        A_s1=np.hstack([np.zeros((Np,Np*m)),I_n,np.zeros((Np,Np))]); A_s2=np.hstack([np.zeros((Np,Np*m)),np.zeros((Np,Np)),I_n])
        A_slew=np.hstack([self.Dd,np.zeros((Np*m,2*Np))])
        self.A_c=sparse.csc_matrix(np.vstack([A_box,A_nh,A_tn,A_s1,A_s2,A_slew]))
    def _dist(self,Dseq):
        Np=self.Np; N=self.km.A.shape[0]; gd=np.zeros(Np*4); s=np.zeros(N)
        for i in range(Np): s=self.km.A@s+self.km.Bd@Dseq[i]; gd[i*4:(i+1)*4]=self.km.C@s
        return gd
    def solve(self, x, Dseq, slew=(45.0, 0.6)):
        Np=self.Np; m=self.m; z=self.km.lift(x).ravel(); y_meas=outputs_of(x).ravel()
        if self.y_pred_next is not None:
            innov=y_meas-self.y_pred_next; self.bias+=0.5*innov
            self.innov_nh_var=0.9*self.innov_nh_var+0.1*innov[1]**2
        b=self.bias
        bo_nh=self.backoff_nh0+(1.5*np.sqrt(self.innov_nh_var) if self.dynamic_backoff else 0.0)
        bo_nh=float(np.clip(bo_nh,self.backoff_nh0,2.5))
        gd=self._dist(Dseq)
        c_do=self.Phi_do@z+gd[self.iDO]+b[0]-self.do_ref
        c_nh=self.Phi_nh@z+gd[self.iNH]+b[1]; c_tn=self.Phi_tn@z+gd[self.iTN]+b[2]
        c_n2o=self.Phi_n2o@z+gd[self.iN2O]+b[3]
        e0=np.zeros(Np*m); e0[0]=self.u_prev[0]; e0[1]=self.u_prev[1]
        q_base=2*self.q_do*self.Gu_do.T@c_do + self.w_air*self.sel_air/KLA_MAX \
               + self.w_pump*self.sel_pump/QA_MAX - 2*self.Dd.T@np.diag(np.tile(self.r_du,Np))@e0
        # bounds
        big=1e12
        lo=np.tile([0.0,0.0],Np); hi=np.tile([KLA_MAX,QA_MAX],Np)
        u_nh=(NH_LIM-bo_nh)-c_nh; u_tn=(TN_LIM-self.backoff_tn)-c_tn
        l_nh=-big*np.ones(Np); l_tn=-big*np.ones(Np); l_s=np.zeros(2*Np); u_s=big*np.ones(2*Np)
        sl=np.tile([slew[0],slew[1]*PM.P['Q0']],Np); l_sl=-sl.copy(); u_sl=sl.copy()
        l_sl[0]=-sl[0]+self.u_prev[0]; u_sl[0]=sl[0]+self.u_prev[0]
        l_sl[1]=-sl[1]+self.u_prev[1]; u_sl[1]=sl[1]+self.u_prev[1]
        l=np.concatenate([lo,l_nh,l_tn,l_s[:Np],l_s[Np:],l_sl]); u=np.concatenate([hi,u_nh,u_tn,u_s[:Np],u_s[Np:],u_sl])
        # --- SLP on the emission term: E_i = K_EM*KLa_i*y_N2O_i ---
        U0=self.warm[:Np*m] if self.warm is not None else np.tile(self.u_prev,Np)
        solve_ms=0.0; Uopt=np.tile(self.u_prev,Np)
        for _ in range(self.slp_iters):
            kla0=U0[self.kla_pos]                       # K_La trajectory at linearization pt
            y0=c_n2o+self.Gu_n2o@U0                     # predicted dissolved N2O there
            g=self.w_N*K_EM*(self.Gu_n2o.T@kla0)        # d/dU of sum K_La0_i*y_N2O_i(U)
            g[self.kla_pos]+=self.w_N*K_EM*y0           # d/dU of sum y_N2O0_i*K_La_i
            q=np.concatenate([q_base+g, self.rho*np.ones(2*Np)])
            if self._prob is None:
                self._prob=osqp.OSQP()
                self._prob.setup(self.P,q,self.A_c,l,u,**osqp_settings(30000))
            else:
                self._prob.update(q=q,l=l,u=u)
            t0=time.perf_counter(); res=self._prob.solve(); solve_ms+=(time.perf_counter()-t0)*1e3
            if res.info.status_val in (1,2): Uopt=res.x[:Np*m]; self.warm=res.x; U0=Uopt.copy()
        u0=np.array([np.clip(Uopt[0],0,KLA_MAX), np.clip(Uopt[1],0,QA_MAX)])
        self.y_pred_next=np.array([
            self.Phi_do[0]@z+self.Gu_do[0]@Uopt+gd[self.iDO][0]+b[0],
            self.Phi_nh[0]@z+self.Gu_nh[0]@Uopt+gd[self.iNH][0]+b[1],
            self.Phi_tn[0]@z+self.Gu_tn[0]@Uopt+gd[self.iTN][0]+b[2],
            self.Phi_n2o[0]@z+self.Gu_n2o[0]@Uopt+gd[self.iN2O][0]+b[3]])
        self.u_prev=u0; return u0, solve_ms, bo_nh

# ------------------------- conventional MIMO baseline: cascade PI -------------------------
class CascadePI:
    """Ammonia-based DO-setpoint outer loop (-> inner DO PI on K_La) plus a
    nitrate/TN-based loop on the internal recirculation Q_a. Fast inner loop 1 min,
    supervisory loops every 15 min. This is the standard multi-loop plant practice."""
    def __init__(self, dt_pi=1.0/60/24, dt_sup=15.0/60/24,
                 Kp=25.0, Ki=90.0, Kabac=0.55, NH_tgt=2.5, DO_min=1.0, DO_max=3.5,
                 Ktn=0.35, TN_tgt=10.0, Qa_min=0.5, Qa_max=4.0):
        self.dt=dt_pi; self.dt_sup=dt_sup; self.Kp=Kp; self.Ki=Ki; self.I=0.0
        self.Kabac=Kabac; self.NH_tgt=NH_tgt; self.DO_min=DO_min; self.DO_max=DO_max
        self.Ktn=Ktn; self.TN_tgt=TN_tgt; self.Qa_min=Qa_min; self.Qa_max=Qa_max
        self.do_sp=DO_min; self.Qa=2.0*PM.P['Q0']; self.KLa=120.0; self._tacc=1e9
    def step(self, x):
        e=PM.effluent(x); self._tacc+=self.dt
        if self._tacc>=self.dt_sup:                     # supervisory update
            self.do_sp=float(np.clip(self.DO_min+self.Kabac*(e['S_NH']-self.NH_tgt),self.DO_min,self.DO_max))
            qaf=float(np.clip(1.0+self.Ktn*(e['N_tot']-self.TN_tgt),self.Qa_min,self.Qa_max))
            self.Qa=qaf*PM.P['Q0']; self._tacc=0.0
        err=self.do_sp-e['DO']; self.I+=err*self.dt
        u=self.Kp*err+self.Ki*self.I; kla=float(np.clip(u,0,KLA_MAX))
        if kla!=u: self.I-=err*self.dt
        self.KLa=kla; return np.array([kla,self.Qa])

# ------------------------- closed-loop runners -------------------------
def run_mpc(x0, mpc, days=12.0, delay_steps=1, forecast="noisy", fc_sigma=0.20,
            meas_noise=0.02, rng_=None, plant_scale=None):
    if rng_ is None: rng_=np.random.default_rng(0)
    n=int(days/DT); x=x0.copy(); Np=mpc.Np; buf=[x.copy() for _ in range(delay_steps+1)]
    delayed=[PM.CO['S_NH'], PM.CO['S_N2O']]
    log=_newlog()
    for k in range(n):
        t=k*DT; S_S,NH_in,X_S,X_BH,Q=PM.influent(t)
        xs=x.copy().reshape(PM.NTANK,PM.NCOMP); xd=buf[0].reshape(PM.NTANK,PM.NCOMP)
        for c in delayed: xs[:,c]=xd[:,c]
        xs=xs.reshape(-1)
        if meas_noise>0: xs=xs*(1.0+meas_noise*rng_.standard_normal(PM.NX))
        if forecast=="persistence":
            Dseq=np.tile([NH_in,Q/PM.P['Q0']],(Np,1))
        else:
            Dseq=np.array([[PM.influent(t+i*DT)[1],PM.influent(t+i*DT)[4]/PM.P['Q0']] for i in range(Np)])
            if forecast=="noisy" and fc_sigma>0:
                err=1.0+fc_sigma*np.sqrt(np.arange(Np))[:,None]*rng_.standard_normal((Np,2))
                Dseq=Dseq*np.clip(err,0.3,2.0)
        u,ms,bo=mpc.solve(xs,Dseq)
        _logstep(log,t,u,x,ms,plant_scale)
        # The plant is integrated with the same 0.5-minute sub-step as the baseline
        # runner. The 3.75-minute sub-step used previously (nsub=4 at DT=15 min) does
        # not resolve the oxygen mode or the N2O stripping term at high K_L a, and made
        # the MPC and the baseline two different numerical experiments.
        x=PM.rk4_step(x,list(u),(S_S,NH_in,X_S,X_BH,Q),DT,nsub=SUBSTEPS(DT),scale=plant_scale)
        buf.append(x.copy()); buf.pop(0)
    return _finalize(log),x

def run_baseline(x0, ctrl, days=12.0, plant_scale=None):
    dt=ctrl.dt; n=int(days/dt); x=x0.copy(); log=_newlog()
    for k in range(n):
        t=k*dt; S_S,NH_in,X_S,X_BH,Q=PM.influent(t); u=ctrl.step(x)
        _logstep(log,t,u,x,0.0,plant_scale)
        x=PM.rk4_step(x,list(u),(S_S,NH_in,X_S,X_BH,Q),dt,nsub=SUBSTEPS(dt),scale=plant_scale)
    return _finalize(log),x

def _newlog(): return dict(t=[],KLa=[],Qa=[],DO=[],NH=[],TN=[],N2O_em=[],ms=[])
def _logstep(log,t,u,x,ms,scale):
    em,_=PM.emission_rate(x,u,scale=scale); e=PM.effluent(x)
    log['t'].append(t);log['KLa'].append(u[0]);log['Qa'].append(u[1]/PM.P['Q0'])
    log['DO'].append(e['DO']);log['NH'].append(e['S_NH']);log['TN'].append(e['N_tot'])
    log['N2O_em'].append(em);log['ms'].append(ms)
def _finalize(log):
    for k in log: log[k]=np.array(log[k])
    return log

def metrics(log):
    dt=float(log['t'][1]-log['t'][0]); days=log['t'][-1]-log['t'][0]
    ae=np.trapezoid(PM.P['So_sat']/(1.8*1000.0)*VAER_TOT*log['KLa'],dx=dt)/days
    pe=np.trapezoid(0.004*log['Qa']*PM.P['Q0'],dx=dt)/days
    n2o=np.trapezoid(log['N2O_em'],dx=dt)/1000.0/days
    # Nitrogen-load effluent-quality index in the style of the BSM Effluent Quality
    # Index, with equal nitrogen weights beta_TKN = beta_NO = 20; the solids, COD and
    # BOD terms of the benchmark EQI are omitted because the ideal point settler does
    # not resolve them, so this is an index of the same form but not the benchmark
    # EQI itself, and it is reported as such. OCI = aeration + pumping + anoxic
    # mixing (+ constant sludge term omitted as it is control-invariant here).
    Q=PM.P['Q0']; TKN=log['NH']; NO=np.maximum(log['TN']-log['NH'],0.0)
    eqi=np.trapezoid((20.0*TKN+20.0*NO)*Q,dx=dt)/(days*1000.0)     # kg pollutant-eq / d
    mix=24.0*0.005*float(np.sum(PM.VOL[PM.ANOX]))                  # anoxic mixing (kWh/d)
    oci=ae+pe+mix
    return dict(AE_kWh_d=float(ae),PE_kWh_d=float(pe),N2O_kgN_d=float(n2o),
                EQI=float(eqi),OCI_kWh_d=float(oci),
                NH_mean=float(np.mean(log['NH'])),NH_peak=float(np.max(log['NH'])),
                NH_viol_h=float(np.sum(log['NH']>NH_LIM)*dt*24),
                TN_mean=float(np.mean(log['TN'])),TN_peak=float(np.max(log['TN'])),
                TN_viol_h=float(np.sum(log['TN']>TN_LIM)*dt*24),
                DO_mean=float(np.mean(log['DO'])),
                ms_mean=float(np.mean(log['ms'][log['ms']>0]) if np.any(log['ms']>0) else 0.0),
                ms_max=float(np.max(log['ms']) if np.any(log['ms']>0) else 0.0))

def identify(x_ss, seed=7, days=12.0):
    global rng; rng=np.random.default_rng(seed)
    X,U,D,Xp=generate_data(x_ss,days=days); ntr=int(0.7*len(X)); Y=outputs_of(X)
    return Koopman(X[:ntr]).fit(X[:ntr],U[:ntr],D[:ntr],Xp[:ntr],Y[:ntr]), (X,U,D,Xp,ntr)

# ============================================================================
if __name__=="__main__":
    import sys
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    if stage in ("ident","all"):
        print("settling ASM1 5-tank plant..."); x_ss=settle(); np.save('x_ss_cascade.npy',x_ss)
        e=PM.effluent(x_ss); print("  eff DO=%.2f NH=%.2f TN=%.2f"%(e['DO'],e['S_NH'],e['N_tot']))
        km,(X,U,D,Xp,ntr)=identify(x_ss,seed=7)
        km.save('koop_cascade.npz')
        Hp=8; sc=np.array([3,10,12,0.12]); ek=[]
        for s in range(ntr,len(X)-Hp,25):
            yk=km.predict_multi(X[s],U[s:s+Hp],D[s:s+Hp]); yt=outputs_of(Xp[s:s+Hp]); ek.append(((yk-yt)/sc)**2)
        rk=np.sqrt(np.mean(ek,0))
        print("  lift N=%d NRMSE@6=%.3f (DO/NH/TN/N2O=%s)"%(km.N,np.mean(rk[5]),np.round(rk[5],2)))
        print("STAGE_IDENT_DONE")
    if stage in ("control","all"):
        x_ss=np.load('x_ss_cascade.npy'); km=Koopman.load('koop_cascade.npz')
        print("MIMO Koopman-MPC (emission-SLP) vs cascade-PI baseline...")
        mpc=KoopMPC(km,w_N=W_N_OPERATING)
        lk,_=run_mpc(x_ss,mpc,days=12.0,rng_=np.random.default_rng(3)); mk=metrics(lk)
        lb,_=run_baseline(x_ss,CascadePI(),days=12.0); mb=metrics(lb)
        for nm,mv in [("Koopman",mk),("CascadePI",mb)]:
            print("  %-9s DO=%.2f AE=%.0f PE=%.0f N2O=%.2f NHpk/viol=%.2f/%.1f TNmean/viol=%.2f/%.1f solve=%.1f/%.1fms"%(
                nm,mv['DO_mean'],mv['AE_kWh_d'],mv['PE_kWh_d'],mv['N2O_kgN_d'],mv['NH_peak'],mv['NH_viol_h'],
                mv['TN_mean'],mv['TN_viol_h'],mv['ms_mean'],mv['ms_max']))
        json.dump(dict(Koopman=mk,CascadePI=mb),open('results_cascade.json','w'),indent=2)
        print("STAGE_CONTROL_DONE")
