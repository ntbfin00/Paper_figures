import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fs = 10 

mock_cov = np.load("vsf_mock_cov.npy")[::-1, ::-1]
poisson_cov = np.load("vsf_poisson_cov.npy")[::-1, ::-1]

def cov2corr(cov):
    std = np.sqrt(np.diag(cov))

    std[std == 0] = np.inf

    R = cov / np.outer(std, std)
    return R

vmax = max(mock_cov.max(), poisson_cov.max())
vmin = min(mock_cov.min(), poisson_cov.min())

plot_kwargs = {'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1, 
               'aspect': 'equal', 'origin': 'lower'}

fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(7,2.5))

# plot correlation matrix 
im = ax1.imshow(cov2corr(mock_cov), **plot_kwargs)
# plot covariance diagonals
r = np.load("../Figure2/vsf.npy")[0,0,::-1]
ax2.plot(r, np.diag(mock_cov), label="Mock covariance")
ax2.plot(r, np.diag(poisson_cov), label="'Poisson' covariance", color='k', ls='--')
ax2.legend(fontsize=3/4*fs)
ax1.set_xlabel('$R_v$ [$h^{-1}\, \mathrm{Mpc}$]', fontsize=fs)
ax2.set_xlabel('$R_v$ [$h^{-1}\, \mathrm{Mpc}$]', fontsize=fs)
ax1.set_ylabel('$R_v$ [$h^{-1}\, \mathrm{Mpc}$]', fontsize=fs)
ax2.set_ylabel('$\mathsf{C}_{ii}$ [$h^{6}\, \mathrm{Mpc}^{-6}$]', fontsize=fs)

# set correlation matrix ticks
ticks = range(len(r))[::2]
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels([f'{rr:.0f}' for rr in r[::2]])
ax1.set_yticklabels([f'{rr:.0f}' for rr in r[::2]])
ax1.tick_params(axis='both', labelsize=fs * 3/4)
ax2.tick_params(axis='both', labelsize=fs * 3/4)

# add colorbar
cbar = fig.colorbar(im, ax=ax1, location='right')
# cbar.set_label(r'$\mathsf{C}_{ij}$', fontsize=fs)
cbar.set_ticks(np.linspace(-1,1,5))
cbar.ax.tick_params(labelsize=fs * 3/4)

# plt.tight_layout(w_pad=0.)
fig.subplots_adjust(wspace=0.3)
plt.savefig("Figure3.pdf", bbox_inches='tight')
plt.show()

