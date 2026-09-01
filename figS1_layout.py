"""Supporting Information Figure S1: five-tank cascade layout and manipulated inputs."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import plant_model_cascade as PM

fig, ax = plt.subplots(figsize=(10.5, 4.0)); ax.set_xlim(0,1); ax.set_ylim(-0.02,1); ax.axis('off')
x0, w, gap, y, h = 0.135, 0.093, 0.012, 0.42, 0.20
names = ['anoxic 1','anoxic 2','aerated 3','aerated 4','aerated 5']
for i,(nm,V) in enumerate(zip(names, PM.VOL)):
    x = x0 + i*(w+gap)
    fc = '#e2e8f0' if i < 2 else '#bfdbfe'
    ax.add_patch(Rectangle((x,y), w, h, fc=fc, ec='#475569', lw=1.3, zorder=3))
    ax.text(x+w/2, y+h*0.62, nm, ha='center', fontsize=8.2, zorder=4)
    ax.text(x+w/2, y+h*0.28, f'{V:g} m$^3$', ha='center', fontsize=7.4, color='#475569', zorder=4)
    if i >= 2:
        ax.add_patch(FancyArrowPatch((x+w/2, y-0.11), (x+w/2, y), arrowstyle='-|>',
                                     mutation_scale=11, lw=1.2, color='#2b6cb0', zorder=4))
xs = x0 + 5*(w+gap)
ax.add_patch(Rectangle((xs,y), 0.075, h, fc='#f1f5f9', ec='#475569', lw=1.3, zorder=3))
ax.text(xs+0.0375, y+h/2, 'point\nsettler', ha='center', va='center', fontsize=7.6, zorder=4)

ax.add_patch(FancyArrowPatch((0.045,y+h/2), (x0,y+h/2), arrowstyle='-|>', mutation_scale=13, lw=1.6, color='#1a202c'))
ax.text(0.043, y+h/2+0.055, 'influent\n$S_{NH}^{in}$, $S_S$, $X_S$, $Q_0$', ha='left', fontsize=8)
ax.add_patch(FancyArrowPatch((xs+0.075,y+h/2), (0.955,y+h/2), arrowstyle='-|>', mutation_scale=13, lw=1.6, color='#1a202c'))
ax.text(0.955, y+h/2+0.05, 'effluent', ha='right', fontsize=8)

ax.text(0.5, y-0.155, '$K_La$ applied to the three aerated tanks   (0–260 d$^{-1}$)',
        ha='center', fontsize=8.6, color='#2b6cb0')

ax.add_patch(FancyArrowPatch((x0+4*(w+gap)+w/2, y+h), (x0+w/2, y+h), arrowstyle='-|>',
                             mutation_scale=12, lw=1.5, color='#0f766e',
                             connectionstyle="arc3,rad=0.30", zorder=5))
ax.text(0.5, y+h+0.20, 'internal nitrate recirculation $Q_a$  (second manipulated input)',
        ha='center', fontsize=8.6, color='#0f766e')
# return activated sludge: settler underflow back to the head of the train
yr = y - 0.235
ax.plot([xs+0.0375, xs+0.0375], [y, yr], color='#7c3aed', lw=1.2, zorder=2)
ax.plot([xs+0.0375, x0+w/2], [yr, yr], color='#7c3aed', lw=1.2, zorder=2)
ax.add_patch(FancyArrowPatch((x0+w/2, yr), (x0+w/2, y), arrowstyle='-|>',
                             mutation_scale=11, lw=1.2, color='#7c3aed', zorder=2))
ax.text(xs-0.02, yr-0.045, 'return activated sludge', ha='right', fontsize=7.8, color='#7c3aed')
ax.text(0.5, 0.02, 'Eight ASM1 components per tank across five tanks: 40 states. '
        'Sensors on effluent ammonia and N$_2$O carry a 15-minute delay.',
        ha='center', fontsize=7.8, color='#475569')
plt.tight_layout(); plt.savefig('figS1_cascade_layout.png', dpi=300, bbox_inches='tight'); plt.close()
print('figS1_cascade_layout.png saved')
