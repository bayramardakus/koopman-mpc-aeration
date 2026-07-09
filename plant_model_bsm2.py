"""
plant_model_bsm2.py  (v2 - ASM1/BSM1-calibrated)
Five-tank BSM2/BSM1-style activated-sludge cascade with N2O, recalibrated to the
official IWA ASM1 stoichiometry and kinetics so that the aerated zones carry a
realistic oxygen demand (DO no longer saturates at peak load).

Key change vs. the earlier reduced model: slowly-biodegradable COD (X_S) is
carried explicitly and HYDROLYSED to readily-biodegradable substrate (S_S)
throughout the train (ASM1 process 7). Hydrolysis sustains heterotrophic oxygen
uptake in the aerated tanks, giving a physically realistic DO profile, and it
keeps enough carbon in the anoxic zone for the internal recycle Q_a to retain
authority over effluent total nitrogen.

Layout (BSM1 bioreactor):
    influent -> [T1 anoxic][T2 anoxic][T3 aer][T4 aer][T5 aer] -> settler
                    ^                                    |
                    |------------ internal recycle Q_a --|
                    ^                                              |
                    |------------------ RAS Q_r ------- settler underflow

Manipulated inputs (MIMO): u = [K_La, Q_a]  (K_La applied to aerated tanks only).

State: 5 tanks x 8 components = 40 states.
    per tank c = [S_O, S_S, S_NH, S_NO, X_S, X_BH, X_BA, S_N2O]
    x[i*8 + c]  for tank i=0..4, component c=0..7

REFERENCE IMPLEMENTATION using ASM1 kinetics with an ideal point-settler; the N2O
submodel is the two-pathway AOB + heterotrophic-denitrification structure of
Section 2. Kinetic/stoichiometric defaults are the BSM1 (15 C) values.
"""
import numpy as np

NTANK = 5
NCOMP = 8
NX = NTANK * NCOMP                       # 40 states
AER = np.array([2, 3, 4])                # aerated tanks
ANOX = np.array([0, 1])
CO = dict(S_O=0, S_S=1, S_NH=2, S_NO=3, X_S=4, X_BH=5, X_BA=6, S_N2O=7)
PART = (CO['X_S'], CO['X_BH'], CO['X_BA'])   # particulates (settle)

# ---- ASM1 / BSM1 parameters (15 C defaults) ----
P = dict(
    muH=4.0, K_S=10.0, K_OH=0.50, K_NO=0.5, b_H=0.30, Y_H=0.67, eta_g=0.80, eta_h=0.80,
    muA=0.70, K_NH=1.0, K_OA=1.00, b_A=0.05, Y_A=0.24,
    i_XB=0.08, f_P=0.08, k_h=3.0, K_X=0.10, So_sat=8.0,
    # N2O submodel (two AOB pathways + heterotrophic-denitrification leak)
    k_n2o_aob=0.10, K_O_n2o=0.5, k_n2o_hao=0.012, K_O_hao=1.5,
    k_n2o_den=0.06, alpha_n2o=0.55,
    # hydraulics / geometry (BSM1)
    V_anox=1000.0, V_aer=1333.0,
    Q0=18446.0, Q_r=18446.0, Q_w=150.0,
    settler_capture=0.995,
)
VOL = np.array([P['V_anox'], P['V_anox'], P['V_aer'], P['V_aer'], P['V_aer']])
Q_A_MAX = 5.0 * P['Q0']

# tunable influent characteristics (BSM1-like fractions)
INF = dict(S_S=90.0, X_S=320.0, X_BH=30.0, S_NH=30.0, rain_Q=1.5, storm_Q=1.8)

def _params(scale):
    return P if scale is None else {k:(P[k]*scale.get(k,1.0)) for k in P}

def influent(t):
    """Influent to Tank1: (S_S, S_NH, X_S, X_BH, Q0) with diurnal + weather."""
    diu = 1.0 + 0.35*np.sin(2*np.pi*(t-0.30))
    Q = P['Q0']*(1.0+0.15*np.sin(2*np.pi*(t-0.30)))
    S_S=INF['S_S']*diu; S_NH=INF['S_NH']*diu; X_S=INF['X_S']*diu; X_BH=INF['X_BH']*diu
    if 5.0 <= t < 9.0:                     # rain
        Q *= INF['rain_Q']; f=0.66; S_S*=f; S_NH*=f; X_S*=f; X_BH*=0.7
    elif 9.0 <= t < 11.0:                  # storm
        Q *= INF['storm_Q']; S_S*=0.85; S_NH*=0.60; X_S*=0.85; X_BH*=0.6
    return S_S, S_NH, X_S, X_BH, Q

def _react(c, KLa, Pp):
    """ASM1 (+N2O) per-unit-volume rates for one tank. c = 8-vector."""
    S_O,S_S,S_NH,S_NO,X_S,X_BH,X_BA,S_N2O = np.maximum(c, 1e-9)
    mon_S=S_S/(Pp['K_S']+S_S); mon_OH=S_O/(Pp['K_OH']+S_O); inh_OH=Pp['K_OH']/(Pp['K_OH']+S_O)
    mon_NO=S_NO/(Pp['K_NO']+S_NO); mon_NH=S_NH/(Pp['K_NH']+S_NH); mon_OA=S_O/(Pp['K_OA']+S_O)
    r1=Pp['muH']*mon_S*mon_OH*X_BH                          # aerobic het growth
    r2=Pp['muH']*mon_S*inh_OH*mon_NO*Pp['eta_g']*X_BH       # anoxic het growth (denit)
    r3=Pp['muA']*mon_NH*mon_OA*X_BA                         # autotroph growth (nitrif)
    r4=Pp['b_H']*X_BH; r5=Pp['b_A']*X_BA                    # decay
    Xr=X_S/max(X_BH,1e-6)
    r6=Pp['k_h']*(Xr/(Pp['K_X']+Xr))*(mon_OH+Pp['eta_h']*inh_OH*mon_NO)*X_BH   # hydrolysis
    # N2O
    low_do=Pp['K_O_n2o']/(Pp['K_O_n2o']+S_O); high_do=S_O/(Pp['K_O_hao']+S_O)
    nitrif=r3/Pp['Y_A']
    r_n2o_aob=(Pp['k_n2o_aob']*low_do+Pp['k_n2o_hao']*high_do)*nitrif
    denit=(1.0-Pp['Y_H'])/(2.86*Pp['Y_H'])*r2
    r_n2o_den=Pp['k_n2o_den']*denit
    r_strip=Pp['alpha_n2o']*KLa*S_N2O
    dc=np.zeros(NCOMP)
    dc[CO['S_O']]  = KLa*(Pp['So_sat']-S_O) - (1-Pp['Y_H'])/Pp['Y_H']*r1 - (4.57-Pp['Y_A'])/Pp['Y_A']*r3
    dc[CO['S_S']]  = -(1.0/Pp['Y_H'])*(r1+r2) + r6
    dc[CO['S_NH']] = -Pp['i_XB']*(r1+r2) - (Pp['i_XB']+1.0/Pp['Y_A'])*r3 + Pp['i_XB']*(r4+r5)
    dc[CO['S_NO']] = (1.0/Pp['Y_A'])*r3 - (1-Pp['Y_H'])/(2.86*Pp['Y_H'])*r2
    dc[CO['X_S']]  = (1.0-Pp['f_P'])*(r4+r5) - r6
    dc[CO['X_BH']] = r1 + r2 - r4
    dc[CO['X_BA']] = r3 - r5
    dc[CO['S_N2O']]= r_n2o_aob + r_n2o_den - r_strip
    return dc

def derivs(x, u, dist, scale=None):
    Pp=_params(scale); KLa, Q_a = u
    S_S,S_NH,X_S,X_BH,Q0 = dist
    Q_r=Pp['Q_r']; Q_w=Pp['Q_w']; cap=Pp['settler_capture']
    Qth = Q0 + Q_a + Q_r
    X = x.reshape(NTANK, NCOMP); C5 = X[4]
    # ideal point-settler on the (Q0+Q_r) stream leaving tank 5
    Qf=Q0+Q_r; Q_e=Q0-Q_w; Q_u=Q_r+Q_w
    C_ras=C5.copy()
    for cc in PART:
        X_eff=C5[cc]*(1.0-cap); C_ras[cc]=(C5[cc]*Qf - X_eff*Q_e)/Q_u
    C5_rec=C5.copy(); C5_rec[CO['S_O']]=0.0; C_ras[CO['S_O']]=0.0   # de-oxygenated recycle
    C0=np.zeros(NCOMP)
    C0[CO['S_S']]=S_S; C0[CO['S_NH']]=S_NH; C0[CO['X_S']]=X_S; C0[CO['X_BH']]=X_BH
    dx=np.zeros((NTANK,NCOMP))
    for i in range(NTANK):
        KLa_i = KLa if i in AER else 0.0
        Fin = (Q0*C0 + Q_a*C5_rec + Q_r*C_ras) if i==0 else Qth*X[i-1]
        dx[i] = (Fin - Qth*X[i])/VOL[i] + _react(X[i], KLa_i, Pp)
    return dx.reshape(-1)

def effluent(x):
    C5=x.reshape(NTANK,NCOMP)[4]
    return dict(DO=C5[CO['S_O']], S_NH=C5[CO['S_NH']],
                N_tot=C5[CO['S_NH']]+C5[CO['S_NO']], S_N2O=C5[CO['S_N2O']])

def emission_rate(x, u, scale=None):
    """Total fugitive N2O stripping emission from aerated tanks (gN/d), dissolved sum."""
    Pp=_params(scale); KLa=u[0]; X=x.reshape(NTANK,NCOMP)
    em=0.0; diss=0.0
    for i in AER:
        s=max(X[i,CO['S_N2O']],0.0); em+=Pp['alpha_n2o']*KLa*s*VOL[i]; diss+=s
    return em, diss

def outputs(x):
    """[DO(tank5), S_NH_eff, N_tot_eff, N2O_dissolved_aer(volume-weighted proxy)]."""
    e=effluent(x); X=x.reshape(NTANK,NCOMP)
    n2o = sum(VOL[i]*X[i,CO['S_N2O']] for i in AER)/np.sum(VOL[AER])
    return np.array([e['DO'], e['S_NH'], e['N_tot'], n2o])

def rk4_step(x, u, dist, dt, nsub=6, scale=None):
    h=dt/nsub
    for _ in range(nsub):
        k1=derivs(x,u,dist,scale); k2=derivs(x+0.5*h*k1,u,dist,scale)
        k3=derivs(x+0.5*h*k2,u,dist,scale); k4=derivs(x+h*k3,u,dist,scale)
        x=x+(h/6.0)*(k1+2*k2+2*k3+k4); x=np.maximum(x,0.0)
    return x

def initial_state():
    x=np.zeros((NTANK,NCOMP))
    for i in range(NTANK):
        x[i]=[1.5 if i in AER else 0.05, 20.0, 5.0, 8.0, 60.0, 2000.0, 120.0, 0.03]
    return x.reshape(-1)

if __name__=="__main__":
    dt=15.0/60/24
    print("ASM1-calibrated 5-tank: DO/NH/TN vs K_La (Q_a=2*Q0):")
    print(f"{'KLa':>5} {'DO5':>6} {'NH':>7} {'NO':>7} {'TN':>7} {'N2O(kgN/d)':>11}")
    for KLa in [60,120,180,240,320]:
        x=initial_state()
        for _ in range(int(35/dt)): x=rk4_step(x,[KLa,2*P['Q0']],influent(3.0),dt,nsub=4)
        a=[]
        for _ in range(int(3/dt)):
            x=rk4_step(x,[KLa,2*P['Q0']],influent(3.0),dt,nsub=4); e=effluent(x); em,_=emission_rate(x,[KLa,2*P['Q0']])
            a.append([e['DO'],e['S_NH'],e['N_tot']-e['S_NH'],e['N_tot'],em/1000])
        a=np.mean(a,0); print(f"{KLa:5d} {a[0]:6.2f} {a[1]:7.3f} {a[2]:7.2f} {a[3]:7.2f} {a[4]:11.2f}")
    print("\nQ_a authority over TN (KLa=180):")
    for qaf in [1.0,2.0,3.0,4.0]:
        x=initial_state()
        for _ in range(int(35/dt)): x=rk4_step(x,[180,qaf*P['Q0']],influent(3.0),dt,nsub=4)
        a=[]
        for _ in range(int(3/dt)):
            x=rk4_step(x,[180,qaf*P['Q0']],influent(3.0),dt,nsub=4); e=effluent(x); a.append([e['DO'],e['S_NH'],e['N_tot']])
        a=np.mean(a,0); print(f"  Qa/Q0={qaf:.1f}  DO={a[0]:.2f} NH={a[1]:.2f} TN={a[2]:.2f}")
