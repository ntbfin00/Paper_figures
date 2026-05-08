import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fs = 10  # fontsize
ms = 5  # markersize
lw = 1.5  # linewidth

# load evaluation times
times = np.load("timings.npy")
ngrid = [400, 500, 600]
nthreads = times[0,0]

cmap = plt.get_cmap("plasma")
colors = cmap(np.linspace(0., 0.8, len(ngrid)))

# plot evaluation times against number of threads
fig, ax = plt.subplots(1, figsize=(10/3,2.5))
for (i,t) in enumerate(times[::-1]):
    ax.plot(*t, label=f'$N_\mathrm{{cells}}={ngrid[::-1][i]}^3$', 
            lw=lw, color=colors[i],
            marker='o', markersize=ms, markerfacecolor='none')
ax.set_ylabel("Void-finding time [s]", fontsize=fs)
ax.set_xlabel("Number of threads", fontsize=fs)
ax.legend(fontsize=fs * 3/4)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks(nthreads)
ax.set_xticklabels([str(n) for n in nthreads])
ax.tick_params(axis='both', which='major', labelsize=fs * 3/4)
ax.tick_params(axis='both', which='minor', labelsize=fs * 3/4)
ax.yaxis.grid(True, which='minor')
ax.grid(lw=1., alpha=0.3)

plt.savefig("Figure6.pdf", bbox_inches="tight")
plt.show()
