"""
eval_bsm2.py
Robustness and variance evaluation for the MIMO Koopman-MPC on the ASM1-calibrated
5-tank cascade (koopman_mpc_bsm2.py):

  (A) multi-seed study: re-identify the Koopman predictor and re-run closed-loop
      control over 5 random seeds; report mean +/- std of aeration energy, pumping
      energy, N2O emission, NH4 violation time, and TN violation time.

  (B) model-plant mismatch: the nominal Koopman predictor (nominal weights) controls
      a plant whose kinetic / N2O parameters are perturbed by +/-15-30%.

Usage:  python eval_bsm2.py variance   |   python eval_bsm2.py mismatch
"""
import sys, json, numpy as np
import plant_model_bsm2 as PM
import koopman_mpc_bsm2 as K

DAYS = 7.0
KEYS = ['AE_kWh_d','PE_kWh_d','N2O_kgN_d','NH_viol_h','TN_viol_h','NH_peak','DO_mean']

def variance(seeds=(1,2,3,4,5)):
    x_ss = K.settle(days=28.0)
    rows=[]
    for sd in seeds:
        km,_ = K.identify(x_ss, seed=sd, days=12.0)            # re-identify per seed
        mpc = K.KoopMPC(km, w_N=1.0)
        log,_ = K.run_mpc(x_ss, mpc, days=DAYS, rng_=np.random.default_rng(100+sd))
        m = K.metrics(log); rows.append([m[k] for k in KEYS])
        print("  seed %d: AE=%.0f PE=%.0f N2O=%.2f NHviol=%.1f TNviol=%.1f"%(
            sd,m['AE_kWh_d'],m['PE_kWh_d'],m['N2O_kgN_d'],m['NH_viol_h'],m['TN_viol_h']))
    a=np.array(rows)
    stats={KEYS[i]:dict(mean=float(a[:,i].mean()),std=float(a[:,i].std())) for i in range(len(KEYS))}
    print("\n  MEAN +/- STD over %d seeds:"%len(seeds))
    for k in KEYS: print("    %-12s %8.2f +/- %.2f"%(k,stats[k]['mean'],stats[k]['std']))
    json.dump(stats, open('results_bsm2_variance.json','w'), indent=2)
    return stats

def mismatch():
    x_ss = K.settle(days=28.0)
    km,_ = K.identify(x_ss, seed=7, days=12.0)                 # nominal predictor
    scen = {
        'nominal'        : None,
        'AOB N2O +25%'   : {'k_n2o_aob':1.25,'k_n2o_hao':1.25},
        'muH -20%'       : {'muH':0.80},
        'muA -15%'       : {'muA':0.85},
        'alpha_N2O +30%' : {'alpha_n2o':1.30},
        'K_OA +30% / muA -15%': {'K_OA':1.30,'muA':0.85},
        'combined'       : {'k_n2o_aob':1.2,'muA':0.85,'muH':0.85,'alpha_n2o':1.2},
    }
    out=[]
    for name,sc in scen.items():
        mpc=K.KoopMPC(km, w_N=1.0)                             # nominal weights, nominal model
        log,_=K.run_mpc(x_ss, mpc, days=DAYS, plant_scale=sc, rng_=np.random.default_rng(300))
        m=K.metrics(log)
        out.append(dict(scenario=name, AE=m['AE_kWh_d'], PE=m['PE_kWh_d'], N2O=m['N2O_kgN_d'],
                        NH_viol_h=m['NH_viol_h'], TN_viol_h=m['TN_viol_h'], NH_peak=m['NH_peak'], DO=m['DO_mean']))
        print("  %-20s AE=%.0f N2O=%.2f NHviol=%.1f TNviol=%.1f NHpk=%.2f DO=%.2f"%(
            name,m['AE_kWh_d'],m['N2O_kgN_d'],m['NH_viol_h'],m['TN_viol_h'],m['NH_peak'],m['DO_mean']))
    json.dump(out, open('results_bsm2_mismatch.json','w'), indent=2)
    return out

def frontier():
    """Iso-energy / Pareto comparison: sweep each controller's aggressiveness and
    trace aeration-energy vs ammonia-violation (and N2O). Fair comparison that does
    not rely on a single baseline tuning."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x_ss=K.settle(days=25.0); km,_=K.identify(x_ss,seed=7,days=10.0)
    days=5.0
    print("Koopman-MPC frontier (sweep ammonia back-off):")
    kf=[]
    for bo in [0.0,0.7,1.5,2.2]:
        mpc=K.KoopMPC(km,w_N=1.0,backoff_nh=bo,dynamic_backoff=False)
        lg,_=K.run_mpc(x_ss,mpc,days=days,rng_=np.random.default_rng(3)); m=K.metrics(lg)
        kf.append(m); print("  bo=%.1f AE=%.0f NHviol=%.1f N2O=%.2f"%(bo,m['AE_kWh_d'],m['NH_viol_h'],m['N2O_kgN_d']))
    print("Cascade-PI frontier (sweep ammonia target / DO ceiling):")
    cf=[]
    for nh_tgt,do_max in [(3.5,2.5),(2.0,3.2),(1.0,4.0),(0.4,4.5)]:
        cp=K.CascadePI(NH_tgt=nh_tgt,DO_max=do_max)
        lg,_=K.run_baseline(x_ss,cp,days=days); m=K.metrics(lg)
        cf.append(m); print("  NHtgt=%.1f DOmax=%.1f AE=%.0f NHviol=%.1f N2O=%.2f"%(nh_tgt,do_max,m['AE_kWh_d'],m['NH_viol_h'],m['N2O_kgN_d']))
    json.dump(dict(koopman=kf,cascade=cf),open('results_bsm2_frontier.json','w'),indent=2)
    # figure: AE vs NH violation, and AE vs N2O
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    kAE=[m['AE_kWh_d'] for m in kf]; kNH=[m['NH_viol_h'] for m in kf]; kN=[m['N2O_kgN_d'] for m in kf]
    cAE=[m['AE_kWh_d'] for m in cf]; cNH=[m['NH_viol_h'] for m in cf]; cN=[m['N2O_kgN_d'] for m in cf]
    ax[0].plot(kAE,kNH,'o-',color='#1f77b4',label='Koopman-MPC'); ax[0].plot(cAE,cNH,'s--',color='#d62728',label='Cascade PI')
    ax[0].set_xlabel('Aeration energy (kWh/d)');ax[0].set_ylabel('NH$_4$ violation time (h / 7 d)');ax[0].legend();ax[0].grid(alpha=.3)
    ax[0].set_title('Energy vs ammonia compliance')
    ax[1].plot(kAE,kN,'o-',color='#1f77b4',label='Koopman-MPC'); ax[1].plot(cAE,cN,'s--',color='#d62728',label='Cascade PI')
    ax[1].set_xlabel('Aeration energy (kWh/d)');ax[1].set_ylabel('N$_2$O emission (kgN/d)');ax[1].legend();ax[1].grid(alpha=.3)
    ax[1].set_title('Energy vs N$_2$O')
    plt.tight_layout();plt.savefig('fig_bsm2_pareto.png',dpi=140);plt.close(); print("fig_bsm2_pareto.png saved")

if __name__=="__main__":
    stage=sys.argv[1] if len(sys.argv)>1 else "variance"
    if stage=="variance":
        print("=== (A) multi-seed variance (re-identify + closed-loop) ==="); variance()
    elif stage=="mismatch":
        print("=== (B) model-plant mismatch (nominal predictor) ==="); mismatch()
    elif stage=="frontier":
        print("=== (C) iso-energy / Pareto comparison ==="); frontier()
