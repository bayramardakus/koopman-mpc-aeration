"""
Reduced ASM1-style activated-sludge model with N2O production, for a single
aerated CSTR with biomass retention (sludge recycle represented by an SRT term).

REFERENCE IMPLEMENTATION -- NOT the official IWA BSM2. It is a physically
motivated reduced surrogate used to generate genuinely-computed (not fabricated)
numbers for a Koopman-MPC control study. All figures/tables produced from it must
be cross-validated against the official BSM2/BSM2-N2O before journal submission.

States (g/m3, i.e. mg/L):
    0 S_O   dissolved oxygen
    1 S_NH  ammonia nitrogen
    2 S_NO  nitrate+nitrite nitrogen
    3 S_S   readily biodegradable COD
    4 X_BH  heterotrophic biomass
    5 X_BA  autotrophic (nitrifier) biomass
    6 S_N2O dissolved nitrous-oxide nitrogen

Input:
    KLa  oxygen transfer coefficient (1/d), the manipulated variable.
Measured disturbances:
    S_NH_in, S_S_in, Q (influent), vary in time (diurnal + dry/rain/storm).
"""
import numpy as np

NX = 7
IDX = dict(S_O=0, S_NH=1, S_NO=2, S_S=3, X_BH=4, X_BA=5, S_N2O=6)

# ---- kinetic / stoichiometric parameters (ASM1-like, ~15 C) ----
P = dict(
    muH=4.0, K_S=10.0, K_OH=0.20, K_NO=0.5, b_H=0.30, Y_H=0.67, eta_g=0.80,
    muA=0.62, K_NH=1.0, K_OA=0.40, b_A=0.06, Y_A=0.24,
    i_XB=0.086,           # N content of biomass (gN/gCOD)
    So_sat=8.0,           # DO saturation (gO2/m3)
    # N2O submodel (two AOB pathways -> U-shaped N2O vs DO)
    k_n2o_aob=0.10,       # low-DO nitrifier-denitrification leak (max)
    K_O_n2o=0.5,          # DO half-sat for the low-DO N2O enhancement (gO2/m3)
    k_n2o_hao=0.012,      # high-activity hydroxylamine pathway (rises with DO, mild)
    K_O_hao=1.5,          # DO half-sat for the high-DO pathway (gO2/m3)
    k_n2o_den=0.06,       # fraction of denitrification flux leaking as N2O
    alpha_n2o=0.55,       # KLa_N2O / KLa_O2 mass-transfer ratio (stripping)
    # reactor
    V=1333.0,             # aerated volume (m3)
    SRT=12.0,             # solids retention time (d)  -> biomass dilution 1/SRT
)

def influent(t):
    """Return (S_NH_in, S_S_in, Q) at time t (days). 14-day dry/rain/storm profile."""
    # diurnal factor (peak midday)
    diu = 1.0 + 0.35 * np.sin(2*np.pi*(t - 0.30))
    base_Q = 18446.0      # m3/d
    NH_in = 25.0 * diu
    S_in = 60.0 * diu
    Q = base_Q * (1.0 + 0.15*np.sin(2*np.pi*(t-0.30)))
    if 5.0 <= t < 9.0:          # rain: higher flow, diluted
        Q *= 1.9
        NH_in *= 0.62; S_in *= 0.62
    elif 9.0 <= t < 11.0:       # storm: brief high-flow first-flush
        Q *= 2.6
        NH_in *= 0.55; S_in *= 0.80
    return NH_in, S_in, Q

def _params(scale):
    return P if scale is None else {k:(P[k]*scale.get(k,1.0)) for k in P}

def derivs(x, KLa, dist, scale=None):
    Pp=_params(scale)
    S_O, S_NH, S_NO, S_S, X_BH, X_BA, S_N2O = x
    S_O=max(S_O,1e-6); S_NH=max(S_NH,1e-9); S_NO=max(S_NO,1e-9); S_S=max(S_S,1e-9)
    X_BH=max(X_BH,1e-9); X_BA=max(X_BA,1e-9); S_N2O=max(S_N2O,0.0)
    NH_in, S_in, Q = dist
    V=Pp['V']; D=Q/V; ds=1.0/Pp['SRT']

    # process rates
    mon_S  = S_S/(Pp['K_S']+S_S)
    mon_OH = S_O/(Pp['K_OH']+S_O)
    ana_OH = Pp['K_OH']/(Pp['K_OH']+S_O)
    mon_NO = S_NO/(Pp['K_NO']+S_NO)
    mon_NH = S_NH/(Pp['K_NH']+S_NH)
    mon_OA = S_O/(Pp['K_OA']+S_O)

    r_h_aer = Pp['muH']*mon_S*mon_OH*X_BH
    r_h_ana = Pp['muH']*Pp['eta_g']*mon_S*ana_OH*mon_NO*X_BH
    r_a     = Pp['muA']*mon_NH*mon_OA*X_BA
    r_dec_H = Pp['b_H']*X_BH
    r_dec_A = Pp['b_A']*X_BA

    # N2O production (two AOB pathways + het denitrification)
    low_do = Pp['K_O_n2o']/(Pp['K_O_n2o']+S_O)
    high_do = S_O/(Pp['K_O_hao']+S_O)
    nitrif_flux = r_a / Pp['Y_A']
    r_n2o_aob = (Pp['k_n2o_aob']*low_do + Pp['k_n2o_hao']*high_do)*nitrif_flux
    denit_flux = r_h_ana*(1.0-Pp['Y_H'])/(2.86*Pp['Y_H'])
    r_n2o_den = Pp['k_n2o_den']*denit_flux
    r_n2o_strip = Pp['alpha_n2o']*KLa*S_N2O

    dx = np.zeros(NX)
    OUR_h = (1.0-Pp['Y_H'])/Pp['Y_H']*r_h_aer
    dx[IDX['S_O']] = KLa*(Pp['So_sat']-S_O) - OUR_h - (4.57-Pp['Y_A'])/Pp['Y_A']*r_a + D*(0.0-S_O)
    dx[IDX['S_NH']] = D*(NH_in-S_NH) - Pp['i_XB']*(r_h_aer+r_h_ana) - (1.0/Pp['Y_A']+Pp['i_XB'])*r_a
    dx[IDX['S_NO']] = D*(0.0-S_NO) + (1.0/Pp['Y_A'])*r_a - (1.0-Pp['Y_H'])/(2.86*Pp['Y_H'])*r_h_ana
    dx[IDX['S_S']] = D*(S_in-S_S) - (1.0/Pp['Y_H'])*(r_h_aer+r_h_ana)
    dx[IDX['X_BH']] = r_h_aer + r_h_ana - r_dec_H - ds*X_BH
    dx[IDX['X_BA']] = r_a - r_dec_A - ds*X_BA
    dx[IDX['S_N2O']] = r_n2o_aob + r_n2o_den - r_n2o_strip + D*(0.0-S_N2O)
    return dx

def emission_rate(x, KLa, scale=None):
    """Actual N2O emission (stripping) flux, gN/m3/d, and mass rate gN/d."""
    Pp=_params(scale)
    S_N2O = max(x[IDX['S_N2O']], 0.0)
    r = Pp['alpha_n2o']*KLa*S_N2O
    return r, r*Pp['V']

def rk4_step(x, KLa, dist, dt, nsub=15, scale=None):
    h = dt/nsub
    for _ in range(nsub):
        k1=derivs(x,KLa,dist,scale); k2=derivs(x+0.5*h*k1,KLa,dist,scale)
        k3=derivs(x+0.5*h*k2,KLa,dist,scale); k4=derivs(x+h*k3,KLa,dist,scale)
        x = x + (h/6.0)*(k1+2*k2+2*k3+k4)
        x = np.maximum(x, 0.0)
    return x

def outputs(x):
    """Control/report outputs: DO, S_NH, TN, S_N2O(dissolved)."""
    return np.array([x[IDX['S_O']], x[IDX['S_NH']],
                     x[IDX['S_NH']]+x[IDX['S_NO']], x[IDX['S_N2O']]])

if __name__ == "__main__":
    # steady-state sanity + constant-KLa sweep to confirm the tradeoff
    x0 = np.array([2.0, 5.0, 8.0, 5.0, 1500.0, 120.0, 0.05])
    dt = 15.0/60/24
    print("constant-KLa sweep (settle 40 d then average 4 d):")
    print(f"{'KLa':>6} {'DO':>6} {'S_NH':>7} {'TN':>7} {'N2O_em(kgN/d)':>13} {'energy(rel)':>11}")
    for KLa in [40,80,120,160,200,240,320]:
        x=x0.copy()
        for _ in range(int(40/dt)):
            t=0.0; x=rk4_step(x,KLa,influent(3.0),dt)  # fixed mid-dry influent
        acc=[]
        for _ in range(int(4/dt)):
            x=rk4_step(x,KLa,influent(3.0),dt)
            _,em=emission_rate(x,KLa); acc.append([x[0],x[1],x[1]+x[2],em/1000.0])
        a=np.mean(acc,0)
        print(f"{KLa:6.0f} {a[0]:6.2f} {a[1]:7.3f} {a[2]:7.2f} {a[3]:13.3f} {KLa/240.0:11.2f}")
