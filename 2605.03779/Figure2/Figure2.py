import numpy as np
import matplotlib.pyplot as plt

fs = 10  # fontsize
ms = 3  # markersize
lw = 1.5  # linewidth

labels = ['Merge all', 'Merge >10% overlap', 'Merge > 20% overlap', 'No overlap']
markers = ['o', 'v', 's', 'p']

cmap = plt.get_cmap("plasma")
colors = cmap(np.linspace(0., 0.8, len(labels)))

fig, ax = plt.subplots(1, figsize=(10/3,3))
for (i,merge) in enumerate(['merge_all', 'merge_90', 'merge_80', 'merge_none']):
    Nv = np.load(f"Nv_versus_{merge}.npy")[:2]
    ax.plot(*Nv, label=labels[i], color=colors[i],
            lw=lw, marker='o', markersize=ms)

ax.set_yscale('log')
ax.set_ylim(0.5, 4e3)
ax.set_ylabel('$N_v$', fontsize=fs)
ax.set_xlabel('$R_v$ [$h^{-1}$Mpc]', fontsize=fs)
ax.tick_params(axis='both', which='major', labelsize=fs * 3/4)
ax.tick_params(axis='both', which='minor', labelsize=fs * 3/4)
ax.legend(loc='lower left', fontsize=fs * 0.65)
ax.grid(lw=1., alpha=0.3)

plt.tight_layout()
plt.savefig("Figure2.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()
