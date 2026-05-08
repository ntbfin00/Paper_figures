import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# fig = plt.figure(figsize=(8,4))
fig = plt.figure(figsize=(7,3.5))

gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.6])

ax1 = fig.add_subplot(gs[0, 0])  # top-left
ax2 = fig.add_subplot(gs[1, 0])  # bottom-left
ax3 = fig.add_subplot(gs[:, 1])  # right, spans both rows
axes = np.array([ax1, ax2, ax3])
plt.setp(ax1.get_xticklabels(), visible=False)

colors = plt.get_cmap("Paired").colors
fs = 10  # fontsize
ms = 3  # markersize
lw = 1.5  # linewidth

biases = [3.2, 2.9]
bias_vide_match = 2
delta_v = -0.8

Nv = {}
# plot density profiles
for (i,vf) in enumerate(['versus', 'vide_cleaned']):
    bias = biases[i]
    append = "" if i > 0 else "_merge_90"
    Nv[vf] = np.load(f"Nv_{vf}{append}.npy")
    for (j,tracer) in enumerate(['g', 'dm']):
        delta = np.load(f"delta_{tracer}_{vf}.npy")
        axes[i].plot(*delta, marker=['o','v'][j], 
                     markersize=ms, lw=lw, 
                     c=colors[4*i+(1-j)])
    axes[i].text(0.98, 0., ["VERSUS", "VIDE"][i],
                 transform=axes[i].transAxes,
                 weight='semibold',
                 ha="right", va="bottom")
    axes[i].plot(delta[0], bias * delta[1], lw=lw,
                 ls='--', c='k')
    axes[i].text(0.98, 0.78, f"$b_v={bias}$",
                 transform=axes[i].transAxes,
                 ha="right", va="bottom")
    axes[i].set_ylabel("$\Delta(<r)$", fontsize=fs)
    axes[i].set_xlim(0, 4)
    axes[i].set_ylim(-1, 0.05)
    axes[i].tick_params(axis='both', which='major', labelsize=fs * 3/4)
    axes[i].tick_params(axis='both', which='minor', labelsize=fs * 3/4)
    axes[i].grid(lw=1., alpha=0.3)
    axes[i].hlines(delta_v, 0, 5, ls=':', color='k', alpha=0.7)
    axes[i].vlines(1, -1, 0.05, ls=':', color='k', alpha=0.7)
ax2.set_xlabel("$r / R_v$", fontsize=fs)

# create legend
labels = ['$\delta_g$', '$\delta_m$', '$b_v \delta_{m}$']
legend_lines = [
    Line2D([0], [0], color='dimgrey', linestyle='-', 
           marker='o', ms=ms),
    Line2D([0], [0], color='darkgrey', linestyle='-', 
           marker='v', ms=ms),
    Line2D([0], [0], color='black', linestyle='--')
]
ax1.legend(legend_lines, labels, loc='right', fontsize=fs * 3/4)

# plot void size function
Nv['theory_versus'] = np.load(f"Nv_theory_bias_{biases[0]}.npy")
Nv['theory_vide'] = np.load(f"Nv_theory_bias_{biases[1]}.npy")
Nv['theory_vide_match'] = np.load(f"Nv_theory_bias_{bias_vide_match}.npy")
ax3.plot(*Nv['theory_versus'], c='k', ls='--')
ax3.plot(*Nv['theory_vide'], c=colors[4], ls='--')
ax3.plot(*Nv['theory_vide_match'], c=colors[4], ls=':')
ax3.errorbar(*Nv['versus'], c=colors[1], 
             ls='none', marker='o', markersize=ms)
ax3.errorbar(*Nv['vide_cleaned'], c=colors[5], 
             ls='none', marker='o', markersize=ms)
ax3.text(0.9, 0.58, f"$b_v={biases[0]}$",
         transform=ax3.transAxes, ha="right", va="bottom")
ax3.text(0.68, 0.45, f"$b_v={biases[1]}$", color=colors[5], alpha=0.7,
         transform=ax3.transAxes, ha="right", va="bottom")
ax3.text(0.7, 0.2, f"$b_v={bias_vide_match}$", color=colors[5], alpha=0.7,
         transform=ax3.transAxes, ha="right", va="bottom")
ax3.set_yscale('log')
ax3.set_ylim(0.5, 4e3)
ax3.set_ylabel('$N_v$', fontsize=fs)
ax3.set_xlabel('$R_v$ [$h^{-1}$Mpc]', fontsize=fs)
ax3.set_xlim(25, 61)
ax3.tick_params(axis='both', which='major', labelsize=fs * 3/4)
ax3.tick_params(axis='both', which='minor', labelsize=fs * 3/4)

plt.tight_layout(h_pad=0.3)
plt.savefig("Figure8.pdf", bbox_inches="tight")
plt.show()
