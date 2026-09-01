"""
Koopman (EDMD) linear predictor + condensed-QP MPC on the reduced ASM+N2O plant.
Generates real, computed numbers and figures for the WER manuscript.

Stages:
  1. settle plant to quasi-steady state
  2. generate excitation data (PRBS-like KLa over operating range)
  3. EDMD identification (identity + RBF lift, disturbance-aware inputs)
  4. multi-step prediction validation vs local linearization  -> Fig1, Table1
  5. closed-loop Koopman-MPC vs PI baseline                    -> Fig2, Table2
  6. sweep N2O weight -> energy-N2O-effluent Pareto            -> Fig3, Table3, Table4

REFERENCE IMPLEMENTATION -- not the official BSM2. See plant_model.py header.
"""
import numpy as np, time, json
if not hasattr(np,'trapezoid'): np.trapezoid=np.trapz  # numpy 1.x/2.x compat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
import osqp


# --- solver settings (revision): tight tolerances + polishing, so the closed loop
# --- is invariant to the OSQP build. See "Computational environment" in the paper.
def osqp_settings(max_iter=20000):
    s = dict(verbose=False, warm_start=True,
             eps_abs=1e-7, eps_rel=1e-7,
             eps_prim_inf=1e-9, eps_dual_inf=1e-9,
             max_iter=max_iter,
             # OSQP derives its default adaptive-rho interval from a wall-clock
             # setup-time measurement, which makes the iterate sequence machine- and
             # load-dependent. Fixing the interval makes every solve deterministic.
             adaptive_rho=True, adaptive_rho_interval=50)
    s['polish' if osqp.__version__.startswith('0.') else 'polishing'] = True
    return s

R_DU = 3e-1   # move-suppression weight (set by tuning study)
import plant_model as PM

rng = np.random.default_rng(7)
DT = 15.0/60/24            # control interval = 15 min (days)
KLA_MAX = 360.0
NH_LIM, TN_LIM = 4.0, 18.0
DO_REF = 2.0

# state normalization scales (typical magnitudes)
SCALE = np.array([3.0, 8.0, 12.0, 15.0, 1600.0, 130.0, 0.15])

def settle(KLa=150.0, days=60.0):
    x = np.array([2.0,5.0,8.0,5.0,1500.0,120.0,0.05])
    n=int(days/DT)
    for _ in range(n):
        x=PM.rk4_step(x,KLa,PM.influent(3.0),DT)
    return x

# ----------------------------------------------------------------------------
# 1-2. Excitation data
# ----------------------------------------------------------------------------
def generate_data(x0, days=22.0):
    n=int(days/DT); X=[];U=[];D=[];Xp=[]
    x=x0.copy(); KLa=150.0; hold=0
    for k in range(n):
        t=k*DT
        if hold<=0:
            KLa=float(np.clip(rng.uniform(60,300)+rng.normal(0,20),20,KLA_MAX))
            hold=rng.integers(2,10)
        hold-=1
        NH_in,S_in,Q=PM.influent(t)
        d=np.array([NH_in,Q/18446.0])       # disturbance inputs (Q normalized)
        X.append(x.copy());U.append([KLa]);D.append(d)
        x=PM.rk4_step(x,KLa,(NH_in,S_in,Q),DT)
        Xp.append(x.copy())
    return map(np.array,(X,U,D,Xp))

# ----------------------------------------------------------------------------
# 3. EDMD lifting + regression
# ----------------------------------------------------------------------------
class Koopman:
    def __init__(self, Xtr, n_rbf=40, ridge=1e-4):
        self.n_rbf=n_rbf
        Xn=Xtr/SCALE
        idx=rng.choice(len(Xn),size=n_rbf,replace=False)
        self.centers=Xn[idx]
        # width from median pairwise distance among centers
        dd=np.linalg.norm(self.centers[:,None,:]-self.centers[None,:,:],axis=2)
        self.sigma=np.median(dd[dd>0])
        self.ridge=ridge
        self.N=NX_lift=NX_l = PM.NX + n_rbf
    def lift(self, X):
        X=np.atleast_2d(X); Xn=X/SCALE
        d2=np.sum((Xn[:,None,:]-self.centers[None,:,:])**2,axis=2)
        rbf=np.exp(-d2/(2*self.sigma**2))
        return np.hstack([Xn, rbf])         # [normalized state | rbf]
    def save(self,path):
        np.savez(path,A=self.A,Bu=self.Bu,Bd=self.Bd,C=self.C,
                 centers=self.centers,sigma=self.sigma,ridge=self.ridge,n_rbf=self.n_rbf)
    @classmethod
    def load(cls,path):
        d=np.load(path); o=cls.__new__(cls)
        o.A=d['A'];o.Bu=d['Bu'];o.Bd=d['Bd'];o.C=d['C']
        o.centers=d['centers'];o.sigma=float(d['sigma']);o.ridge=float(d['ridge'])
        o.n_rbf=int(d['n_rbf']);o.N=o.A.shape[0]
        return o
    def fit(self, X,U,D,Xp,Y):
        Z=self.lift(X); Zp=self.lift(Xp)
        Om=np.hstack([Z,U,D])               # [z | KLa | dist]
        n=Z.shape[1]; m=Om.shape[1]
        G=Om.T@Om + self.ridge*np.eye(m)
        AB=(np.linalg.solve(G, Om.T@Zp)).T   # (n x m)
        self.A=AB[:, :n]; self.Bu=AB[:, n:n+1]; self.Bd=AB[:, n+1:]
        # output matrix C: z -> [DO,NH,TN,N2O]
        Gz=Z.T@Z + self.ridge*np.eye(n)
        self.C=(np.linalg.solve(Gz, Z.T@Y)).T   # (4 x n)
        return self
    def predict_multi(self, x0, Useq, Dseq):
        z=self.lift(x0).ravel(); ys=[]
        for u,d in zip(Useq,Dseq):
            z=self.A@z + self.Bu[:,0]*u + self.Bd@d
            ys.append(self.C@z)
        return np.array(ys)

def outputs_of(X):
    X=np.atleast_2d(X)
    return np.stack([X[:,0],X[:,1],X[:,1]+X[:,2],X[:,6]],axis=1)

# local linearization predictor (finite-diff Jacobian at mean operating point)
def local_linear_model(x_bar,u_bar,d_bar,eps=1e-4):
    f0=PM.rk4_step(x_bar,u_bar,(d_bar[0],60.0,d_bar[1]*18446.0),DT)
    A=np.zeros((PM.NX,PM.NX))
    for j in range(PM.NX):
        dx=np.zeros(PM.NX); dx[j]=eps*max(abs(x_bar[j]),1e-3)
        fp=PM.rk4_step(x_bar+dx,u_bar,(d_bar[0],60.0,d_bar[1]*18446.0),DT)
        A[:,j]=(fp-f0)/dx[j]
    du=eps*u_bar
    fu=PM.rk4_step(x_bar,u_bar+du,(d_bar[0],60.0,d_bar[1]*18446.0),DT)
    B=((fu-f0)/du).reshape(-1,1)
    return A,B,f0,x_bar,u_bar
def local_predict(lin,x0,Useq):
    A,B,f0,xb,ub=lin; x=x0.copy(); ys=[]
    for u in Useq:
        x=f0 + A@(x-xb) + (B[:,0]*(u-ub)); x=np.maximum(x,0)
        ys.append([x[0],x[1],x[1]+x[2],x[6]])
        f0=PM.rk4_step(xb,ub,(0,60,0),DT)*0+f0  # keep affine anchor fixed
    return np.array(ys)

# ----------------------------------------------------------------------------
# 5. Condensed-QP MPC
# ----------------------------------------------------------------------------
class KoopMPC:
    def __init__(self, km, Np=12, w_E=1.0, w_N=1.0, q_do=0.0, r_du=None,
                 rho=8e3, eps_s=1.0, s_n2o=0.10, n2o_mode="emission", slp_iters=2,
                 offset_free=True):
        # Economic MPC. n2o_mode="emission": penalize the ACTUAL fugitive N2O
        # emission alpha*KLa*S_N2O (bilinear in the input and a lifted output),
        # handled by successive linearization (SLP); "dissolved": legacy proxy on
        # S_N2O (kept for the ablation reported in the paper). q_do=0 -> economic.
        self.km=km; self.Np=Np; self.w_E=w_E; self.w_N=w_N; self.s_n2o=s_n2o
        self.q_do=q_do; self.r_du=(R_DU if r_du is None else r_du); self.rho=rho; self.eps_s=eps_s
        self.n2o_mode=n2o_mode; self.slp_iters=slp_iters; self.offset_free=offset_free
        self.alpha_n2o=PM.P['alpha_n2o']
        self._build_static()
        self.u_prev=150.0; self.warm=None
    def _build_static(self):
        A,Bu=self.km.A,self.km.Bu[:,0]; C=self.km.C; N=A.shape[0]; Np=self.Np
        # Phi (Np*4 x N), Gu (Np*4 x Np), and per-step A^i, conv for disturbance
        self.Apow=[np.linalg.matrix_power(A,i) for i in range(Np+1)]
        Phi=np.zeros((Np*4,N)); Gu=np.zeros((Np*4,Np))
        self.CA=[C@self.Apow[i] for i in range(Np+1)]
        for i in range(Np):
            Phi[i*4:(i+1)*4,:]=self.CA[i+1]
            for j in range(i+1):
                Gu[i*4:(i+1)*4,j]=self.CA[i-j]@Bu
        self.Phi=Phi; self.Gu=Gu
        # split rows by output
        self.r_do=slice(0,None,4)  # will index via arrays instead
        idx=lambda o:[i*4+o for i in range(Np)]
        self.iDO,self.iNH,self.iTN,self.iN2O=idx(0),idx(1),idx(2),idx(3)
        self.Gu_do=Gu[self.iDO]; self.Gu_nh=Gu[self.iNH]
        self.Gu_tn=Gu[self.iTN]; self.Gu_n2o=Gu[self.iN2O]
        self.Phi_do=Phi[self.iDO]; self.Phi_nh=Phi[self.iNH]
        self.Phi_tn=Phi[self.iTN]; self.Phi_n2o=Phi[self.iN2O]
        # move matrix (Delta U incl u0-u_prev)
        Dd=np.eye(Np)-np.diag(np.ones(Np-1),-1)
        self.Dd=Dd
        # static P (independent of state)
        self.qN=self.w_N/(self.s_n2o**2)         # scaled dissolved-N2O weight (legacy)
        qN_quad = self.qN if self.n2o_mode=="dissolved" else 0.0  # emission term is LINEAR
        P_UU=2*(self.q_do*self.Gu_do.T@self.Gu_do
                + qN_quad*self.Gu_n2o.T@self.Gu_n2o
                + self.r_du*Dd.T@Dd)
        P=np.block([[P_UU,np.zeros((Np,2*Np))],
                    [np.zeros((2*Np,Np)),2*self.eps_s*np.eye(2*Np)]])
        self.P=sparse.csc_matrix((P+P.T)/2)
        self.Bd=self.km.Bd
        # constraint matrix A_c (static part)
        Np2=Np
        # rows: box U (Np), NH (Np), TN (Np), slack>=0 (2Np), slew (Np)
        I=np.eye(Np)
        A_box=np.hstack([I,np.zeros((Np,2*Np))])
        A_nh=np.hstack([self.Gu_nh,-I,np.zeros((Np,Np))])
        A_tn=np.hstack([self.Gu_tn,np.zeros((Np,Np)),-I])
        A_s1=np.hstack([np.zeros((Np,Np)),I,np.zeros((Np,Np))])
        A_s2=np.hstack([np.zeros((Np,Np)),np.zeros((Np,Np)),I])
        A_slew=np.hstack([Dd,np.zeros((Np,2*Np))])
        self.A_c=sparse.csc_matrix(np.vstack([A_box,A_nh,A_tn,A_s1,A_s2,A_slew]))
        self.Np2=Np2
        self._prob=None   # persistent OSQP instance (setup once)
        self.bias=np.zeros(4)      # offset-free output-disturbance estimate
        self.y_pred_next=None      # last one-step-ahead output prediction
        self.nh_margin=1.0; self.tn_margin=1.5  # compliance back-off
    def dist_contrib(self, Dseq):
        # disturbance-only propagation using a horizon FORECAST Dseq (Np x 2)
        Np=self.Np; N=self.km.A.shape[0]
        gd=np.zeros(Np*4); s=np.zeros(N)
        for i in range(Np):
            s=self.km.A@s + self.Bd@Dseq[i]
            gd[i*4:(i+1)*4]=self.km.C@s
        return gd
    def solve(self, x, Dseq, slew=60.0):
        z=self.km.lift(x).ravel(); Np=self.Np
        # ---- offset-free correction: update output-disturbance bias ----
        y_meas=np.array([x[0],x[1],x[1]+x[2],x[6]])
        if self.y_pred_next is not None and self.offset_free:
            self.bias += 0.5*(y_meas-self.y_pred_next)   # innovation filter
        b=self.bias
        gd=self.dist_contrib(Dseq)
        c_do=self.Phi_do@z+gd[self.iDO]+b[0]-DO_REF
        c_n2o=self.Phi_n2o@z+gd[self.iN2O]+b[3]
        c_nh=self.Phi_nh@z+gd[self.iNH]+b[1]
        c_tn=self.Phi_tn@z+gd[self.iTN]+b[2]
        e0=np.zeros(Np); e0[0]=self.u_prev
        # base gradient (DO tracking + aeration energy + move suppression)
        q_U_base=2*self.q_do*self.Gu_do.T@c_do + (self.w_E/360.0)*np.ones(Np) \
                 - 2*self.r_du*self.Dd.T@e0
        # constraint bounds (constant over the SLP iterations)
        big=1e7
        l_box=np.zeros(Np); u_box=KLA_MAX*np.ones(Np)
        u_nh=(NH_LIM-self.nh_margin)-c_nh; u_tn=(TN_LIM-self.tn_margin)-c_tn
        l_nh=-big*np.ones(Np); l_tn=-big*np.ones(Np)
        l_s=np.zeros(2*Np); u_s=big*np.ones(2*Np)
        l_slew=(-slew*np.ones(Np)); u_slew=slew*np.ones(Np)
        l_slew[0]=-slew+self.u_prev; u_slew[0]=slew+self.u_prev
        l=np.concatenate([l_box,l_nh,l_tn,l_s[:Np],l_s[Np:],l_slew])
        u=np.concatenate([u_box,u_nh,u_tn,u_s[:Np],u_s[Np:],u_slew])
        # --- N2O objective: emission alpha*KLa*S_N2O via successive linearization ---
        u0_vec=self.warm[:Np] if self.warm is not None else self.u_prev*np.ones(Np)
        niter=self.slp_iters if self.n2o_mode=="emission" else 1
        solve_ms=0.0; Uopt=self.u_prev*np.ones(Np)
        for _ in range(niter):
            if self.n2o_mode=="emission":
                y3_0=c_n2o + self.Gu_n2o@u0_vec               # predicted dissolved N2O at U0
                # d/dU sum_i alpha*(u0_i*S_N2O_i(U) + S_N2O_i0*u_i) : linear emission term
                q_n2o=self.w_N*self.alpha_n2o*(self.Gu_n2o.T@u0_vec + y3_0)
            else:
                q_n2o=2*self.qN*self.Gu_n2o.T@c_n2o           # legacy dissolved-N2O proxy
            q=np.concatenate([q_U_base+q_n2o, self.rho*np.ones(2*Np)])
            if self._prob is None:
                self._prob=osqp.OSQP()
                self._prob.setup(self.P,q,self.A_c,l,u,**osqp_settings(20000))
            else:
                self._prob.update(q=q,l=l,u=u)
            t0=time.perf_counter(); res=self._prob.solve(); solve_ms+=(time.perf_counter()-t0)*1e3
            if res.info.status_val in (1,2):
                Uopt=res.x[:Np]; self.warm=res.x; u0_vec=Uopt.copy()
        u0=float(np.clip(Uopt[0],0,KLA_MAX))
        # one-step-ahead output prediction (incl. bias) for next innovation
        self.y_pred_next=np.array([
            self.Phi_do[0]@z + self.Gu_do[0]@Uopt + gd[self.iDO][0] + b[0],
            self.Phi_nh[0]@z + self.Gu_nh[0]@Uopt + gd[self.iNH][0] + b[1],
            self.Phi_tn[0]@z + self.Gu_tn[0]@Uopt + gd[self.iTN][0] + b[2],
            self.Phi_n2o[0]@z + self.Gu_n2o[0]@Uopt + gd[self.iN2O][0] + b[3]])
        self.u_prev=u0
        return u0, solve_ms

# ----------------------------------------------------------------------------
# PI baseline (DO control via KLa)
# ----------------------------------------------------------------------------
class PIController:
    """Fast inner DO loop (BSM-style), run at dt_pi = 1 min."""
    def __init__(self,Kp=25.0,Ki=120.0,setpoint=DO_REF,dt_pi=1.0/60/24):
        self.Kp=Kp;self.Ki=Ki;self.sp=setpoint;self.I=0.0;self.u=150.0;self.dt=dt_pi
    def step(self,x):
        e=self.sp-x[0]; self.I+=e*self.dt
        u=self.Kp*e+self.Ki*self.I
        uc=float(np.clip(u,0,KLA_MAX))
        if uc!=u: self.I-=e*self.dt   # anti-windup
        self.u=uc; return uc

class CascadeABAC:
    """Ammonia-based aeration control (ABAC): supervisory NH->DO setpoint loop
    (updated every dt_sup) cascaded onto a fast inner DO PI (1-min). A modern,
    fair baseline that reacts to the effluent-ammonia signal itself."""
    def __init__(self,Kp=40.0,Ki=200.0,Kabac=0.60,NH_target=2.5,
                 DO_min=1.0,DO_max=4.0,dt_pi=1.0/60/24,dt_sup=15.0/60/24):
        self.inner=PIController(Kp=Kp,Ki=Ki,setpoint=DO_min,dt_pi=dt_pi)
        self.Kabac=Kabac;self.NH_target=NH_target;self.DO_min=DO_min;self.DO_max=DO_max
        self.dt_sup=dt_sup;self.dt_pi=dt_pi;self._tacc=1e9
    def step(self,x):
        self._tacc+=self.dt_pi
        if self._tacc>=self.dt_sup:                 # supervisory setpoint update
            do_sp=self.DO_min+self.Kabac*(x[1]-self.NH_target)
            self.inner.sp=float(np.clip(do_sp,self.DO_min,self.DO_max))
            self._tacc=0.0
        return self.inner.step(x)

def run_pi_fast(x0, controller, days=10.0, meas_noise=0.0, rng=None,
                plant_scale=None):
    """PI/ABAC baseline at 1-min sampling; optional sensor noise & plant mismatch."""
    if rng is None: rng=np.random.default_rng(0)
    dt_pi=1.0/60/24; n=int(days/dt_pi); x=x0.copy()
    log=dict(t=[],KLa=[],DO=[],NH=[],TN=[],N2O_em=[],ms=[])
    for k in range(n):
        t=k*dt_pi; NH_in,S_in,Q=PM.influent(t)
        xm=x*(1.0+meas_noise*rng.standard_normal(PM.NX)) if meas_noise>0 else x
        u=controller.step(xm); _,em=PM.emission_rate(x,u)
        log['t'].append(t);log['KLa'].append(u);log['DO'].append(x[0])
        log['NH'].append(x[1]);log['TN'].append(x[1]+x[2])
        log['N2O_em'].append(em);log['ms'].append(0.0)
        x=PM.rk4_step(x,u,(NH_in,S_in,Q),dt_pi,nsub=2,scale=plant_scale)
    for kk in log: log[kk]=np.array(log[kk])
    return log,x

# ----------------------------------------------------------------------------
# closed-loop runner + metrics
# ----------------------------------------------------------------------------
def run_closed_loop(x0, controller, days=14.0, kind="mpc",
                    forecast="perfect", fc_sigma=0.0, meas_noise=0.0,
                    plant_scale=None, rng=None):
    """forecast: 'perfect' (exact future influent), 'noisy' (multiplicative
    error growing with horizon), or 'persistence' (hold current -> no preview)."""
    if rng is None: rng=np.random.default_rng(0)
    n=int(days/DT); x=x0.copy()
    log=dict(t=[],KLa=[],DO=[],NH=[],TN=[],N2O_em=[],ms=[])
    for k in range(n):
        t=k*DT; NH_in,S_in,Q=PM.influent(t)
        if kind=="mpc":
            Np=controller.Np
            if forecast=="persistence":
                Dseq=np.tile([NH_in,Q/18446.0],(Np,1))
            else:
                Dseq=np.array([[PM.influent(t+i*DT)[0],PM.influent(t+i*DT)[2]/18446.0]
                               for i in range(Np)])
                if forecast=="noisy" and fc_sigma>0:
                    # forecast error grows with lead time (sqrt(i))
                    err=1.0+fc_sigma*np.sqrt(np.arange(Np))[:,None]*rng.standard_normal((Np,2))
                    Dseq=Dseq*np.clip(err,0.3,2.0)
            xm=x*(1.0+meas_noise*rng.standard_normal(PM.NX)) if meas_noise>0 else x
            u,ms=controller.solve(xm,Dseq)
        else:
            u=controller.step(x); ms=0.0
        _,em=PM.emission_rate(x,u,scale=plant_scale)
        log['t'].append(t);log['KLa'].append(u);log['DO'].append(x[0])
        log['NH'].append(x[1]);log['TN'].append(x[1]+x[2])
        log['N2O_em'].append(em);log['ms'].append(ms)
        x=PM.rk4_step(x,u,(NH_in,S_in,Q),DT,scale=plant_scale)
    for kk in log: log[kk]=np.array(log[kk])
    return log,x

def metrics(log):
    dt=float(log['t'][1]-log['t'][0])                      # actual log spacing
    ae=np.trapezoid(PM.P['So_sat']/(1.8*1000.0)*PM.P['V']*log['KLa'],dx=dt)/((log['t'][-1]-log['t'][0]))
    n2o=np.trapezoid(log['N2O_em'],dx=dt)                      # gN over window
    days=log['t'][-1]-log['t'][0]
    n2o_kg_d=n2o/1000.0/days
    nh_viol_h=np.sum(log['NH']>NH_LIM)*dt*24.0            # hours in window
    tn_viol_h=np.sum(log['TN']>TN_LIM)*dt*24.0
    eqi=np.trapezoid(2.0*log['NH']+1.0*(log['TN']-log['NH']),dx=dt)/days  # proxy EQI (N loads)
    return dict(AE_kWh_d=float(ae), N2O_kgN_d=float(n2o_kg_d),
                NH_mean=float(np.mean(log['NH'])), NH_viol_h=float(nh_viol_h),
                TN_viol_h=float(tn_viol_h), NH_peak=float(np.max(log['NH'])),
                EQI_proxy=float(eqi),
                ms_mean=float(np.mean(log['ms'][log['ms']>0]) if np.any(log['ms']>0) else 0.0),
                ms_max=float(np.max(log['ms']) if np.any(log['ms']>0) else 0.0))

# ============================================================================
def stage_ident():
    OUT={}
    print("settling plant..."); x_ss=settle(days=40.0)
    np.save('x_ss.npy',x_ss)
    print("  steady outputs DO=%.2f NH=%.2f TN=%.2f"%(x_ss[0],x_ss[1],x_ss[1]+x_ss[2]))

    print("generating excitation data...")
    X,U,D,Xp=generate_data(x_ss,days=14.0)
    ntr=int(0.7*len(X))
    Y=outputs_of(X)
    km=Koopman(X[:ntr]).fit(X[:ntr],U[:ntr],D[:ntr],Xp[:ntr],Y[:ntr])
    print("  lift dim N=%d (7 state + %d RBF), samples=%d"%(km.N,km.n_rbf,len(X)))

    # ---- 4. multi-step prediction validation on held-out ----
    Hp=12
    starts=range(ntr, len(X)-Hp, 20)
    err_k=np.zeros((4,Hp)); err_lin=np.zeros((4,Hp)); cnt=0
    xb=np.mean(X[:ntr],0); ub=float(np.mean(U[:ntr])); db=np.mean(D[:ntr],0)
    lin=local_linear_model(xb,ub,db)
    for s in starts:
        Useq=U[s:s+Hp,0]; Dseq=D[s:s+Hp]
        yk=km.predict_multi(X[s],Useq,Dseq)
        yl=local_predict(lin,X[s],Useq)
        yt=outputs_of(Xp[s:s+Hp])
        sc=np.array([3.0,8.0,12.0,0.15])
        err_k+=((yk-yt)/sc).T**2; err_lin+=((yl-yt)/sc).T**2; cnt+=1
    rmse_k=np.sqrt(err_k/cnt); rmse_lin=np.sqrt(err_lin/cnt)
    OUT['fig1']=dict(horizon=list(range(1,Hp+1)),
                     rmse_koopman=rmse_k.tolist(), rmse_local=rmse_lin.tolist())
    OUT['table1']=dict(N=km.N,n_rbf=km.n_rbf,n_samples=len(X),train_frac=0.7,
                       ridge=km.ridge,dt_min=15,
                       nrmse_koopman_h6=float(np.mean(rmse_k[:,5])),
                       nrmse_local_h6=float(np.mean(rmse_lin[:,5])),
                       nrmse_koopman_N2O_h6=float(rmse_k[3,5]),
                       nrmse_local_N2O_h6=float(rmse_lin[3,5]))
    print("  NRMSE@6steps  Koopman=%.3f  local=%.3f"%(
        np.mean(rmse_k[:,5]),np.mean(rmse_lin[:,5])))

    # Figure 1
    names=['DO','Effluent NH$_4$','Total N','N$_2$O (dissolved)']
    fig,axs=plt.subplots(1,4,figsize=(15,3.4))
    for o in range(4):
        axs[o].plot(range(1,Hp+1),rmse_k[o],'o-',color='#1f77b4',label='Koopman')
        axs[o].plot(range(1,Hp+1),rmse_lin[o],'s--',color='#d62728',label='Local lin.')
        axs[o].set_title(names[o]);axs[o].set_xlabel('prediction step (15 min)')
        axs[o].grid(alpha=.3)
        if o==0:axs[o].set_ylabel('normalized RMSE');axs[o].legend()
    plt.tight_layout();plt.savefig('fig1_prediction.png',dpi=140);plt.close()
    km.save('koop.npz')
    with open('results_ident.json','w') as f: json.dump(OUT,f,indent=2)
    print("STAGE_IDENT_DONE")

FC_SIGMA=0.20     # influent-forecast error (grows with lead time)
MEAS_NOISE=0.02   # relative sensor noise

def stage_control():
    OUT=json.load(open('results_ident.json'))
    x_ss=np.load('x_ss.npy'); km=Koopman.load('koop.npz')
    # ---- balanced Koopman-MPC (realistic: noisy forecast + sensor noise) ----
    print("closed-loop MPC (balanced, noisy forecast)...")
    mpc=KoopMPC(km,w_E=1.0,w_N=1.0)
    log_mpc,_=run_closed_loop(x_ss,mpc,days=10.0,kind="mpc",forecast="noisy",
                              fc_sigma=FC_SIGMA,meas_noise=MEAS_NOISE,
                              rng=np.random.default_rng(11))
    # ---- baselines: fixed-DO PI and ammonia-based cascade (ABAC) ----
    print("closed-loop PI and ABAC baselines...")
    log_pi,_=run_pi_fast(x_ss,PIController(Kp=40.0,Ki=200.0),days=10.0)
    log_ab,_=run_pi_fast(x_ss,CascadeABAC(),days=10.0)
    m_mpc=metrics(log_mpc); m_pi=metrics(log_pi); m_ab=metrics(log_ab)
    OUT['table2']=dict(MPC=m_mpc,ABAC=m_ab,PI=m_pi)
    for nm,mm in [("MPC",m_mpc),("ABAC",m_ab),("PI",m_pi)]:
        print("  %-4s AE=%.0f N2O=%.2f NHviol=%.1fh NHpk=%.2f"%(
            nm,mm['AE_kWh_d'],mm['N2O_kgN_d'],mm['NH_viol_h'],mm['NH_peak']))

    # ---- forecast-quality sensitivity at the balanced point ----
    print("forecast sensitivity...")
    fcs={}
    for name,fc,sg in [("perfect","perfect",0.0),("noisy","noisy",FC_SIGMA),
                       ("persistence","persistence",0.0)]:
        lg,_=run_closed_loop(x_ss,KoopMPC(km,w_E=1.0,w_N=1.0),days=10.0,kind="mpc",
                             forecast=fc,fc_sigma=sg,meas_noise=MEAS_NOISE,
                             rng=np.random.default_rng(23))
        mm=metrics(lg)
        fcs[name]=dict(AE=mm['AE_kWh_d'],N2O=mm['N2O_kgN_d'],
                       NHviol=mm['NH_viol_h'],NHpk=mm['NH_peak'])
        print("  %-11s AE=%.0f N2O=%.2f NHviol=%.1fh"%(name,mm['AE_kWh_d'],mm['N2O_kgN_d'],mm['NH_viol_h']))
    OUT['forecast_sens']=fcs

    # Figure 2 -- MPC vs ABAC (strong baseline) vs PI
    t=log_mpc['t']; tp=log_pi['t']; ta=log_ab['t']
    fig,axs=plt.subplots(4,1,figsize=(11,9),sharex=True)
    def L(ax,i,key,scale=1.0):
        ax.plot(t,log_mpc[key]*scale,color='#1f77b4',label='Koopman-MPC' if i==0 else None)
        ax.plot(ta,log_ab[key]*scale,color='#2ca02c',lw=.9,alpha=.85,label='ABAC cascade' if i==0 else None)
        ax.plot(tp,log_pi[key]*scale,color='#d62728',lw=.7,alpha=.6,label='PI (DO=2)' if i==0 else None)
    L(axs[0],0,'DO');axs[0].axhline(DO_REF,color='k',ls=':',lw=.8);axs[0].set_ylabel('DO (mg/L)');axs[0].legend(loc='upper right',ncol=3,fontsize=8)
    L(axs[1],1,'KLa');axs[1].set_ylabel('K$_L$a (1/d)')
    L(axs[2],2,'NH');axs[2].axhline(NH_LIM,color='k',ls='--',lw=.8);axs[2].set_ylabel('Effl. NH$_4$ (mgN/L)')
    L(axs[3],3,'N2O_em',1/1000.0);axs[3].set_ylabel('N$_2$O emission (kgN/d)');axs[3].set_xlabel('time (d)')
    for a in axs:a.grid(alpha=.3)
    axs[0].axvspan(5,9,color='#cce5ff',alpha=.3);axs[0].axvspan(9,11,color='#ffe0cc',alpha=.4)
    plt.tight_layout();plt.savefig('fig2_timeseries.png',dpi=140);plt.close()

    # ---- Pareto sweep over w_N (noisy forecast) ----
    print("Pareto sweep...")
    wns=[0.0,0.5,1.0,2.0,4.0,8.0,16.0]; pareto=[]
    for wn in wns:
        lg,_=run_closed_loop(x_ss,KoopMPC(km,w_E=1.0,w_N=wn),days=10.0,kind="mpc",
                             forecast="noisy",fc_sigma=FC_SIGMA,meas_noise=MEAS_NOISE,
                             rng=np.random.default_rng(31))
        mm=metrics(lg); mm['w_N']=wn; pareto.append(mm)
        print("  w_N=%5.2f AE=%.0f N2O=%.2f NHviol=%.1fh NHpk=%.2f"%(
            wn,mm['AE_kWh_d'],mm['N2O_kgN_d'],mm['NH_viol_h'],mm['NH_peak']))
    OUT['pareto']=pareto; OUT['table3']=pareto
    OUT['table4']=dict(
        NH_viol_h_MPC=m_mpc['NH_viol_h'],NH_viol_h_ABAC=m_ab['NH_viol_h'],NH_viol_h_PI=m_pi['NH_viol_h'],
        TN_viol_h_MPC=m_mpc['TN_viol_h'],TN_viol_h_ABAC=m_ab['TN_viol_h'],TN_viol_h_PI=m_pi['TN_viol_h'],
        NH_peak_MPC=m_mpc['NH_peak'],NH_peak_ABAC=m_ab['NH_peak'],NH_peak_PI=m_pi['NH_peak'],
        solve_ms_mean=m_mpc['ms_mean'],solve_ms_max=m_mpc['ms_max'],
        control_interval_ms=15*60*1000)

    # Figure 3 (Pareto) with ABAC and PI reference points
    AE=np.array([p['AE_kWh_d'] for p in pareto]);N2=np.array([p['N2O_kgN_d'] for p in pareto])
    NHv=np.array([p['NH_viol_h'] for p in pareto])
    fig,ax=plt.subplots(figsize=(7.8,5.6))
    sc=ax.scatter(AE,N2,c=NHv,cmap='viridis_r',s=110,zorder=3,edgecolor='k',vmin=0,vmax=16)
    ax.plot(AE,N2,'-',color='gray',alpha=.5,zorder=2)
    for p in pareto: ax.annotate("w$_N$=%g"%p['w_N'],(p['AE_kWh_d'],p['N2O_kgN_d']),textcoords="offset points",xytext=(7,5),fontsize=8)
    ax.scatter([m_pi['AE_kWh_d']],[m_pi['N2O_kgN_d']],marker='*',s=360,color='#d62728',edgecolor='k',zorder=4)
    ax.annotate("PI (%.1f h viol.)"%m_pi['NH_viol_h'],(m_pi['AE_kWh_d'],m_pi['N2O_kgN_d']),textcoords="offset points",xytext=(8,-26),fontsize=9,color='#d62728',fontweight='bold')
    ax.scatter([m_ab['AE_kWh_d']],[m_ab['N2O_kgN_d']],marker='P',s=240,color='#2ca02c',edgecolor='k',zorder=4)
    ax.annotate("ABAC (%.1f h viol.)"%m_ab['NH_viol_h'],(m_ab['AE_kWh_d'],m_ab['N2O_kgN_d']),textcoords="offset points",xytext=(8,8),fontsize=9,color='#2ca02c',fontweight='bold')
    ax.set_xlabel('Aeration energy (kWh/d)');ax.set_ylabel('N$_2$O emission (kgN/d)')
    ax.set_title('Energy-N$_2$O-compliance trade-off')
    cb=plt.colorbar(sc);cb.set_label('effluent NH$_4$ violation (h in 10-d window)')
    ax.grid(alpha=.3);plt.tight_layout();plt.savefig('fig3_pareto.png',dpi=140);plt.close()

    with open('results.json','w') as f: json.dump(OUT,f,indent=2)
    print("STAGE_CONTROL_DONE")

def stage_robust():
    """Multi-seed variability (incl. re-identification) + model-plant mismatch."""
    OUT=json.load(open('results.json')); x_ss=np.load('x_ss.npy')
    # (a) multi-seed: re-identify Koopman and re-run balanced MPC per seed
    print("multi-seed (re-identify + closed-loop)...")
    seeds=[1,2,3,4,5]; rows=[]
    global rng
    for sd in seeds:
        rng=np.random.default_rng(100+sd)
        X,U,D,Xp=generate_data(x_ss,days=14.0); ntr=int(0.7*len(X)); Y=outputs_of(X)
        kms=Koopman(X[:ntr]).fit(X[:ntr],U[:ntr],D[:ntr],Xp[:ntr],Y[:ntr])
        lg,_=run_closed_loop(x_ss,KoopMPC(kms,w_E=1.0,w_N=1.0),days=10.0,kind="mpc",
                             forecast="noisy",fc_sigma=FC_SIGMA,meas_noise=MEAS_NOISE,
                             rng=np.random.default_rng(200+sd))
        mm=metrics(lg); rows.append([mm['AE_kWh_d'],mm['N2O_kgN_d'],mm['NH_viol_h'],mm['NH_peak']])
        print("  seed %d: AE=%.0f N2O=%.2f NHviol=%.1f"%(sd,mm['AE_kWh_d'],mm['N2O_kgN_d'],mm['NH_viol_h']))
    a=np.array(rows); lbl=['AE_kWh_d','N2O_kgN_d','NH_viol_h','NH_peak']
    OUT['multiseed']={lbl[i]:dict(mean=float(a[:,i].mean()),std=float(a[:,i].std())) for i in range(4)}

    # (b) model-plant mismatch: nominal Koopman, perturbed plant
    print("model-plant mismatch...")
    rng=np.random.default_rng(7)
    km=Koopman.load('koop.npz')
    scenarios={'nominal':None,
               'N2O +25%':{'k_n2o_aob':1.25,'k_n2o_hao':1.25},
               'muA -15%':{'muA':0.85},
               'alpha_N2O +30%':{'alpha_n2o':1.30},
               'combined':{'k_n2o_aob':1.2,'muA':0.9,'alpha_n2o':1.2,'K_OA':1.2}}
    mism=[]
    for name,sc in scenarios.items():
        lg,_=run_closed_loop(x_ss,KoopMPC(km,w_E=1.0,w_N=1.0),days=10.0,kind="mpc",
                             forecast="noisy",fc_sigma=FC_SIGMA,meas_noise=MEAS_NOISE,
                             plant_scale=sc,rng=np.random.default_rng(300))
        mm=metrics(lg); mism.append(dict(scenario=name,AE=mm['AE_kWh_d'],N2O=mm['N2O_kgN_d'],
                                         NHviol=mm['NH_viol_h'],NHpk=mm['NH_peak']))
        print("  %-14s AE=%.0f N2O=%.2f NHviol=%.1f NHpk=%.2f"%(name,mm['AE_kWh_d'],mm['N2O_kgN_d'],mm['NH_viol_h'],mm['NH_peak']))
    OUT['mismatch']=mism
    with open('results.json','w') as f: json.dump(OUT,f,indent=2)
    print("STAGE_ROBUST_DONE")

if __name__=="__main__":
    import sys
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    if stage in ("ident","all"): stage_ident()
    if stage in ("control","all"): stage_control()
    if stage in ("robust","all"): stage_robust()
