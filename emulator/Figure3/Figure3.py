import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fs = 10 

mock_cov = np.load("vsf_mock_cov.npy")[::-1, ::-1]
poisson_cov = np.load("vsf_poisson_cov.npy")[::-1, ::-1]

vmax = max(mock_cov.max(), poisson_cov.max())
vmin = min(mock_cov.min(), poisson_cov.min())

plot_kwargs = {'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax, 
               'aspect': 'equal', 'origin': 'lower'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5), sharey=True)

ax1.imshow(mock_cov, **plot_kwargs)
im = ax2.imshow(poisson_cov, **plot_kwargs)
r = np.load("../Figure2/vsf.npy")[0,0,::-1]
ax1.set_ylabel('$R_v$ [$h^{-1}\, \mathrm{Mpc}$]', fontsize=fs)
labels = ["Mock covariance", "'Poisson' covariance"]
for (i,ax) in enumerate([ax1, ax2]):
    ax.set_xlabel('$R_v$ [$h^{-1}\, \mathrm{Mpc}$]', fontsize=fs)
    ticks = range(len(r))[::2]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{rr:.0f}' for rr in r[::2]])
    ax.set_yticklabels([f'{rr:.0f}' for rr in r[::2]])
    ax.tick_params(axis='both', labelsize=fs * 3/4)
    ax.text(
        0.04, 0.96, labels[i],
        transform=ax.transAxes,
        ha='left', va='top', fontsize=fs,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

fig.colorbar(im, ax=[ax1, ax2], location='right',
             label='$\mathsf{C}_{ij}$')

plt.savefig("Figure3.pdf", bbox_inches='tight')

