"""
Independent EVALUATION plant for the cross-model validation of ms. 3395422.

This model exists only to test a controller that was identified on `plant_model.py`.
It is deliberately *structurally* different from that model, not merely differently
parameterised, so that it probes the error class a parameter-perturbation study
cannot reach. Three structural differences:

  1. Ammonia oxidation is resolved in TWO steps with an explicit hydroxylamine
     intermediate S_NH2OH, so the plant carries eight states where the design model
     carries seven. The controller never sees this state.
  2. The AOB nitrifier-denitrification pathway uses a HALDANE oxygen dependence,
     f(S_O) = S_O / (K_O + S_O + S_O^2/K_I), whose maximum lies at sqrt(K_O*K_I) =
     0.65 gO2/m3 (Guo & Vanrolleghem, 2014, ASMG1), in place of the design model's
     inhibition ratio K_O/(K_O+S_O). Both plants are non-monotone in DO, but they
     peak in different places -- the design model near DO = 0.3 gO2/m3, this one
     near DO = 1.0 -- and the pathway partition shifts differently with DO.
  3. A nitrous-oxide reductase term CONSUMES N2O under low-oxygen conditions. The
     design model has no N2O sink at all other than stripping, so this is a
     mechanism it cannot represent.

The N2O parameters are calibrated so that the steady-state emission factor over the
usable aeration range lies at 1.25-1.59% of the influent nitrogen load, inside the
range reported for full-scale facilities (IPCC 2019; Song et al., 2024; Vasilaki et
al., 2019), against 1.45-4.60% for the design model.

Everything not listed above -- ASM1 heterotrophic and autotrophic stoichiometry,
the reactor, the influent series, the aeration actuator and the stripping physics --
is identical to `plant_model.py`, so the comparison isolates the N2O structure.

States (g/m3):
    0 S_O      dissolved oxygen
    1 S_NH     ammonia nitrogen
    2 S_NO     nitrate+nitrite nitrogen (lumped, as in the design model)
    3 S_S      readily biodegradable COD
    4 X_BH     heterotrophic biomass
    5 X_BA     autotrophic (AOB) biomass
    6 S_NH2OH  hydroxylamine nitrogen          <-- not present in the design model
    7 S_N2O    dissolved nitrous-oxide nitrogen
"""
import numpy as np
import plant_model as DESIGN          # shared ASM1 parameters, influent, reactor

NX = 8
IDX = dict(S_O=0, S_NH=1, S_NO=2, S_S=3, X_BH=4, X_BA=5, S_NH2OH=6, S_N2O=7)

# indices of the seven states the controller expects, in its own order
TO_DESIGN = [IDX['S_O'], IDX['S_NH'], IDX['S_NO'], IDX['S_S'],
             IDX['X_BH'], IDX['X_BA'], IDX['S_N2O']]

_D = DESIGN.P
P = dict(
    # --- unchanged ASM1 core, taken from the design model ---
    muH=_D['muH'], K_S=_D['K_S'], K_OH=_D['K_OH'], K_NO=_D['K_NO'],
    b_H=_D['b_H'], Y_H=_D['Y_H'], eta_g=_D['eta_g'],
    muA=_D['muA'], K_NH=_D['K_NH'], K_OA=_D['K_OA'], b_A=_D['b_A'], Y_A=_D['Y_A'],
    i_XB=_D['i_XB'], So_sat=_D['So_sat'], V=_D['V'], SRT=_D['SRT'],
    alpha_n2o=_D['alpha_n2o'],          # same stripping physics

    # --- structural difference 1: two-step AOB with hydroxylamine ---
    k_HAO=6.0,         # hydroxylamine oxidation rate constant (1/d)
    K_NH2OH=0.30,      # half-saturation for NH2OH oxidation (gN/m3)

    # --- structural difference 2: N2O pathways with a Haldane ND term ---
    k_NN=1.15e-2,      # nitrosyl/NH2OH-oxidation leak (gN per gN oxidised)
    K_O_NN=1.0,        # oxygen half-saturation of the NN pathway (gO2/m3)
    k_ND=2.45,         # AOB denitrification leak (1/d)
    K_O_ND=0.4225,     # Haldane affinity  (gO2/m3)
    K_I_ND=1.0,        # Haldane inhibition (gO2/m3); sqrt(K_O*K_I) = 0.65
    k_HD=0.020,        # fraction of heterotrophic denitrification flux leaking

    # --- structural difference 3: N2O reductase (a sink the design model lacks) ---
    k_red=0.13,        # maximum specific N2O reduction rate (1/d per gCOD/m3)
    K_N2O=0.05,        # half-saturation for N2O reduction (gN/m3)
    K_OH_red=0.10,     # oxygen inhibition of N2O reduction (gO2/m3)
)

influent = DESIGN.influent            # identical influent series


def _params(scale):
    return P if scale is None else {k: (P[k] * scale.get(k, 1.0)) for k in P}


def haldane(S_O, Pp):
    """Non-monotone oxygen dependence of the AOB denitrification pathway.
    Maximum at sqrt(K_O_ND*K_I_ND) = 0.65 gO2/m3 (ASMG1)."""
    return S_O / (Pp['K_O_ND'] + S_O + S_O * S_O / Pp['K_I_ND'])


def derivs(x, KLa, dist, scale=None):
    Pp = _params(scale)
    S_O, S_NH, S_NO, S_S, X_BH, X_BA, S_NH2OH, S_N2O = x
    S_O = max(S_O, 1e-6); S_NH = max(S_NH, 1e-9); S_NO = max(S_NO, 1e-9)
    S_S = max(S_S, 1e-9); X_BH = max(X_BH, 1e-9); X_BA = max(X_BA, 1e-9)
    S_NH2OH = max(S_NH2OH, 0.0); S_N2O = max(S_N2O, 0.0)
    NH_in, S_in, Q = dist
    V = Pp['V']; D = Q / V; ds = 1.0 / Pp['SRT']

    mon_S = S_S / (Pp['K_S'] + S_S)
    mon_OH = S_O / (Pp['K_OH'] + S_O)
    ana_OH = Pp['K_OH'] / (Pp['K_OH'] + S_O)
    mon_NO = S_NO / (Pp['K_NO'] + S_NO)
    mon_NH = S_NH / (Pp['K_NH'] + S_NH)
    mon_OA = S_O / (Pp['K_OA'] + S_O)

    r_h_aer = Pp['muH'] * mon_S * mon_OH * X_BH
    r_h_ana = Pp['muH'] * Pp['eta_g'] * mon_S * ana_OH * mon_NO * X_BH
    r_dec_H = Pp['b_H'] * X_BH
    r_dec_A = Pp['b_A'] * X_BA

    # ---- two-step ammonia oxidation ----
    # step 1: NH3 -> NH2OH, oxygen- and ammonia-limited
    r_amo = (Pp['muA'] / Pp['Y_A']) * mon_NH * mon_OA * X_BA
    # step 2: NH2OH -> NO2/NO3, this step carries the growth
    mon_hyd = S_NH2OH / (Pp['K_NH2OH'] + S_NH2OH)
    r_hao = Pp['k_HAO'] * mon_hyd * mon_OA * X_BA
    r_a = Pp['Y_A'] * r_hao                        # autotrophic growth

    # ---- N2O: two AOB pathways with a Haldane ND term, plus HD ----
    r_nn = Pp['k_NN'] * r_hao * S_O / (Pp['K_O_NN'] + S_O)
    r_nd = Pp['k_ND'] * haldane(S_O, Pp) * mon_NO * X_BA / 100.0
    denit_flux = r_h_ana * (1.0 - Pp['Y_H']) / (2.86 * Pp['Y_H'])
    r_hd = Pp['k_HD'] * denit_flux

    # ---- N2O reductase: consumption under low oxygen (absent from the design model)
    r_red = (Pp['k_red'] * S_N2O / (Pp['K_N2O'] + S_N2O)
             * Pp['K_OH_red'] / (Pp['K_OH_red'] + S_O) * X_BH / 1000.0)

    r_strip = Pp['alpha_n2o'] * KLa * S_N2O

    dx = np.zeros(NX)
    OUR_h = (1.0 - Pp['Y_H']) / Pp['Y_H'] * r_h_aer
    dx[IDX['S_O']] = (KLa * (Pp['So_sat'] - S_O) - OUR_h
                      - (4.57 - Pp['Y_A']) / Pp['Y_A'] * r_a + D * (0.0 - S_O))
    dx[IDX['S_NH']] = (D * (NH_in - S_NH) - Pp['i_XB'] * (r_h_aer + r_h_ana)
                       - r_amo - Pp['i_XB'] * r_a)
    dx[IDX['S_NH2OH']] = r_amo - r_hao - D * S_NH2OH
    dx[IDX['S_NO']] = (D * (0.0 - S_NO) + r_hao
                       - (1.0 - Pp['Y_H']) / (2.86 * Pp['Y_H']) * r_h_ana
                       - r_nd)
    dx[IDX['S_S']] = D * (S_in - S_S) - (1.0 / Pp['Y_H']) * (r_h_aer + r_h_ana)
    dx[IDX['X_BH']] = r_h_aer + r_h_ana - r_dec_H - ds * X_BH
    dx[IDX['X_BA']] = r_a - r_dec_A - ds * X_BA
    dx[IDX['S_N2O']] = r_nn + r_nd + r_hd - r_red - r_strip + D * (0.0 - S_N2O)
    return dx


def emission_rate(x, KLa, scale=None):
    """Fugitive N2O emission (stripping) flux, gN/m3/d, and mass rate gN/d."""
    Pp = _params(scale)
    r = Pp['alpha_n2o'] * KLa * max(x[IDX['S_N2O']], 0.0)
    return r, r * Pp['V']


def rk4_step(x, KLa, dist, dt, nsub=15, scale=None):
    h = dt / nsub
    for _ in range(nsub):
        k1 = derivs(x, KLa, dist, scale); k2 = derivs(x + 0.5 * h * k1, KLa, dist, scale)
        k3 = derivs(x + 0.5 * h * k2, KLa, dist, scale); k4 = derivs(x + h * k3, KLa, dist, scale)
        x = np.maximum(x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    return x


def outputs(x):
    """The same four outputs the controller uses. NH2OH is not among them."""
    return np.array([x[IDX['S_O']], x[IDX['S_NH']],
                     x[IDX['S_NH']] + x[IDX['S_NO']], x[IDX['S_N2O']]])


def to_design_state(x):
    """Project the eight-state evaluation state onto the seven-state layout the
    identified predictor expects. The hydroxylamine pool is simply not visible."""
    return np.asarray(x)[TO_DESIGN]


def settle(KLa=150.0, days=60.0, dt=15.0 / 60 / 24):
    x = np.array([2.0, 5.0, 8.0, 5.0, 1500.0, 120.0, 0.05, 0.05])
    for k in range(int(days / dt)):
        x = rk4_step(x, KLa, influent(3.0), dt)
    return x


if __name__ == "__main__":
    # (i) emission factor, (ii) the non-monotone N2O-vs-DO characteristic
    dt = 15.0 / 60 / 24
    print("constant-KLa sweep on the EVALUATION plant (settle 40 d, average 4 d)")
    print(f"{'KLa':>6} {'DO':>6} {'S_NH':>7} {'TN':>7} {'NH2OH':>7} "
          f"{'N2O_em(kgN/d)':>13} {'EF(%)':>7} {'ND(%)':>8}")
    NH_in, S_in, Q = influent(3.0)
    Nload = NH_in * Q / 1000.0
    for KLa in [40, 80, 120, 160, 200, 240, 320]:
        x = settle(KLa, days=40.0)
        acc = []
        for _ in range(int(4 / dt)):
            x = rk4_step(x, KLa, influent(3.0), dt)
            _, em = emission_rate(x, KLa)
            Pp = P
            r_hao_ = Pp['k_HAO']*(x[6]/(Pp['K_NH2OH']+x[6]))*(x[0]/(Pp['K_OA']+x[0]))*x[5]
            nn = Pp['k_NN']*r_hao_*x[0]/(Pp['K_O_NN']+x[0])
            nd = Pp['k_ND']*haldane(x[0],Pp)*(x[2]/(Pp['K_NO']+x[2]))*x[5]/100.0
            acc.append([x[0], x[1], x[1] + x[2], x[6], em / 1000.0, nn, nd])
        a = np.mean(acc, 0)
        frac_nd = 100*a[6]/max(a[5]+a[6],1e-9)
        print(f"{KLa:6.0f} {a[0]:6.2f} {a[1]:7.3f} {a[2]:7.2f} {a[3]:7.4f} "
              f"{a[4]:13.3f} {100*a[4]/Nload:7.2f} {frac_nd:8.1f}")
