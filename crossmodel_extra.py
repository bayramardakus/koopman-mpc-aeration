import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koopman_mpc as K, plant_model as PM, plant_model_eval as EV
import crossmodel as CM

# ---- (a) steady-state emission characteristic of the two plants vs DO ----
dt = 15.0/60/24
NHb,Sb,Qb = EV.influent(3.0); Nload = NHb*Qb/1000.0
rows=[]
for KLa in [40,60,80,100,120,150,180,220,260,320]:
    xd = PM.rk4_step(np.array([2.,5.,8.,5.,1500.,120.,.05]),KLa,EV.influent(3.0),dt)
    for _ in range(int(40/dt)): xd = PM.rk4_step(xd,KLa,EV.influent(3.0),dt)
    xe = EV.settle(KLa, days=40.0)
    ad=[];ae=[]
    for _ in range(int(3/dt)):
        xd = PM.rk4_step(xd,KLa,EV.influent(3.0),dt); _,emd = PM.emission_rate(xd,KLa)
        xe = EV.rk4_step(xe,KLa,EV.influent(3.0),dt); _,eme = EV.emission_rate(xe,KLa)
        ad.append([xd[0],emd/1000.]); ae.append([xe[0],eme/1000.])
    ad=np.mean(ad,0); ae=np.mean(ae,0)
    rows.append(dict(KLa=KLa, DO_design=ad[0], EF_design=100*ad[1]/Nload,
                     DO_eval=ae[0], EF_eval=100*ae[1]/Nload))
    print("KLa %3d | design DO %.2f EF %.2f%% | eval DO %.2f EF %.2f%%"%(
        KLa,ad[0],100*ad[1]/Nload,ae[0],100*ae[1]/Nload), flush=True)
json.dump(rows, open('results_crossmodel_char.json','w'), indent=2)

fig,ax=plt.subplots(figsize=(6.8,4.3))
ax.plot([r['DO_design'] for r in rows],[r['EF_design'] for r in rows],'o-',color='#1f77b4',
        label='design model: two-pathway AOB,\ninhibition-ratio DO law, no N$_2$O sink')
ax.plot([r['DO_eval'] for r in rows],[r['EF_eval'] for r in rows],'s--',color='#d62728',
        label='evaluation model: explicit NH$_2$OH,\nHaldane DO law, N$_2$O reductase')
ax.axhspan(1.0,1.6,color='#94a3b8',alpha=.20,zorder=0)
ax.text(5.35,1.63,'reported full-scale range 1.0-1.6%',fontsize=7.5,color='#475569',
        va='bottom',ha='right')
ax.annotate('peak DO $\\approx$ 0.3',xy=(0.30,4.60),xytext=(1.15,4.35),fontsize=8,color='#1f77b4',
            arrowprops=dict(arrowstyle='->',color='#1f77b4',lw=1))
ax.annotate('peak DO $\\approx$ 1.0',xy=(0.98,1.59),xytext=(1.9,2.25),fontsize=8,color='#d62728',
            arrowprops=dict(arrowstyle='->',color='#d62728',lw=1))
# closed-loop mean DO of the three controllers on the evaluation plant (Table 13)
ax.axvspan(1.0,1.9,color='#22c55e',alpha=.09,zorder=0)
ax.text(1.45,0.42,'closed-loop\nmean DO',fontsize=7.2,color='#15803d',ha='center')
ax.set_xlabel('mean dissolved oxygen (mg O$_2$ L$^{-1}$)')
ax.set_ylabel('emission factor (% of influent N)')
ax.grid(alpha=.3); ax.legend(fontsize=7.8,loc='upper right')
ax.set_title('Steady-state N$_2$O characteristic: design plant versus independent evaluation plant',
             fontsize=9.5)
plt.tight_layout(); plt.savefig('figR10_crossmodel_char.png',dpi=170); plt.close()
print('figR10_crossmodel_char.png saved', flush=True)

# ---- (b) five noise realisations of the unchanged controller on the eval plant ----
x_ss = EV.settle(KLa=150.0, days=40.0)
km = K.Koopman.load('koop.npz')
acc=[]
for sd in range(5):
    lg,_ = CM.run_eval(x_ss, K.KoopMPC(km,w_E=1.0,w_N=1.0), days=10.0, kind='mpc',
                       forecast='noisy', rng=np.random.default_rng(200+sd))
    m = CM._metrics(lg); acc.append([m['AE_kWh_d'],m['N2O_kgN_d'],m['NH_viol_h'],m['NH_peak']])
    print("  seed %d: AE=%.0f N2O=%.2f viol=%.2f pk=%.2f"%(sd,*acc[-1]), flush=True)
a=np.array(acc); lab=['AE_kWh_d','N2O_kgN_d','NH_viol_h','NH_peak']
st={lab[i]:dict(mean=float(a[:,i].mean()),sd=float(a[:,i].std(ddof=1))) for i in range(4)}
json.dump(st, open('results_crossmodel_seeds.json','w'), indent=2)
for k,v in st.items(): print("  %-10s %.2f +/- %.2f"%(k,v['mean'],v['sd']), flush=True)
print('EXTRA_DONE')
