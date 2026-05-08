import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colors = plt.get_cmap("Paired").colors
fs = 10  # fontsize
ms = 3  # markersize
lw = 1.5  # linewidth

Rmax = 2.5

# define analytic density profiles
def density_profile(r, vf):
    if vf == 'versus':
        return 0.84 * np.exp(-5.5e4 * np.exp(-11 * r) - np.exp(-2 * r)) + 0.16
    elif vf == "vide_cleaned":
        return 0.93 / (1 + np.exp(-6.4 * r + 7)) + 0.07

r = np.linspace(0, Rmax, 100)

fig, axes = plt.subplots(2, sharex=True, figsize=(10/3,4), gridspec_kw={'hspace': 0})
for (i,vf) in enumerate(["versus", "vide_cleaned"]):
    delta_abacus = np.load(f"delta_g_abacus_{vf}_r35_40.npy")
    delta_simulated = np.load(f"delta_g_simulated_{vf}.npy")
    axes[i].errorbar(*delta_abacus, color=colors[4*i], lw=lw, 
                     ls='-', marker='v', ms=ms, zorder=1)
    axes[i].plot(r, density_profile(r, vf) - 1, color='k',
                 lw=lw, ls='--', zorder=2)
    axes[i].errorbar(*delta_simulated, color=colors[4*i+1], lw=lw, 
                     ls='-', marker='o', ms=ms, zorder=3)
    axes[i].set_ylabel('$\delta_g$', fontsize=fs)
    axes[i].set_ylim(-1, 0.1)
    axes[i].grid(lw=1., alpha=0.3)
    axes[i].text(0.98, 0., ["VERSUS", "VIDE"][i],
                 transform=axes[i].transAxes,
                 weight='semibold', fontsize=fs,
                 ha="right", va="bottom")
    axes[i].tick_params(axis='both', which='major', labelsize=fs * 3/4)
    axes[i].tick_params(axis='both', which='minor', labelsize=fs * 3/4)
    
axes[1].set_xlabel('$r/R_v$', fontsize=fs)
axes[1].set_xlim(0, Rmax)

# create legend
labels = ["Abacus", "Analytic fit", "Synthetic"]
legend_lines = [
    Line2D([0], [0], color='darkgrey', linestyle='-', 
           marker='v', ms=ms),
    Line2D([0], [0], color='black', linestyle='--'),
    Line2D([0], [0], color='dimgrey', linestyle='-', 
           marker='o', ms=ms),
]
axes[0].legend(legend_lines, labels, loc='right', fontsize=fs * 3/4)

plt.tight_layout()
plt.savefig("Figure4.pdf", bbox_inches="tight")
plt.show()
