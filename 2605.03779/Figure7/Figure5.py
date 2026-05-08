import numpy as np
import matplotlib.pyplot as plt

colors = plt.get_cmap("Paired").colors
fs = 10  # fontsize
ms = 3  # markersize
lw = 1.5  # linewidth

fig, ax = plt.subplots(1, figsize=(10/3,2.5))

# plot true simulated voids
Nv = np.load("Nv_simulated_true.npy")
ax.plot(*Nv, color='grey', lw=lw, ms=ms,
        marker='s', label='Simulated')

Nv = np.insert(Nv, 0, [35, Nv[1,0]], axis=1)
ax.fill_between(*Nv, color='k', alpha=0.1)

# plot Poisson model expectaiton
Nv = np.load("Nv_simulated_model.npy")
ax.plot(*Nv, color='k', lw=lw, ls='--',
        marker='v', ms=ms, label='Prediction')

# plot measured voids
labels = ["VERSUS", "VIDE", "REVOLVER"]
for (i,vf) in enumerate(["versus", "vide_cleaned", "revolver_cleaned"]):
    Nv = np.load(f"Nv_simulated_{vf}.npy")
    ax.errorbar(*Nv, color=colors[[1, 5, 3][i]], label=labels[i],
                marker=['v', 'o', 'p'][i], ms=[ms, ms * 3/4, ms * 3/4][i], 
                lw=1, ls='none', capsize=1.5)

# ax.set_yscale('log')
ax.set_ylabel('$N_v$', fontsize=fs)
ax.set_xlabel('$R_v$ [$h^{-1}$Mpc]', fontsize=fs)
ax.set_xlim(Nv[0][0] - 0.25, Nv[0][-1] + 0.5)
ax.set_ylim(0, 220)
ax.tick_params(axis='both', which='major', labelsize=fs * 3/4)
ax.tick_params(axis='both', which='minor', labelsize=fs * 3/4)
ax.grid(lw=1., alpha=0.3)
ax.legend(fontsize=fs * 3/4)

plt.savefig("Figure7.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()

