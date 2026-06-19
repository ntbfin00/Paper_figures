import numpy as np
import matplotlib.pyplot as plt

fs = 10  # fontsize

def plot_model(ax, x, data, pred, std, c='cadetblue',
               xlabel=None, ylabel=None):
    ax[0].errorbar(x[0], data[0], std[0], c='k',
                   zorder=0, label='Data')
    ax[0].plot(x[0], pred[0], ls='--', c=c,
               zorder=10, label='Model')

    residuals = (data - pred) / std

    # calculate residual percentiles
    one_sigma = np.percentile(residuals, [16, 84], axis=0)
    two_sigma = np.percentile(residuals, [2.5, 97.5], axis=0)
    ax[1].fill_between(x[0], *two_sigma, color=c, alpha=0.4);
    ax[1].fill_between(x[0], *one_sigma, color=c, alpha=0.6);

    xlim = x[0].min(), x[0].max()
    ax[0].set_xlim(*xlim)
    ax[1].set_xlim(*xlim)
    ax[1].set_ylim(-2.5, 2.5)
    ax[0].set_xticks([])
    ax[1].set_yticks([-2,0,2])
    ax[0].tick_params(axis='both', which='major', labelsize=fs * 3/4)
    ax[1].tick_params(axis='both', which='major', labelsize=fs * 3/4)

    if ylabel is not None:
        ax[0].set_ylabel(rf'{ylabel}', fontsize=fs)
        ax[1].set_ylabel(rf'$\Delta {ylabel.split("$")[1]} / \sigma$',
                         fontsize=fs)
    if xlabel is not None:
        ax[1].set_xlabel(rf'{xlabel} [$h^{{-1}}$ Mpc]', fontsize=fs)

vsf = np.load("vsf.npy")
xi = np.load("vg_ccf.npy")
mid = xi.shape[2] // 2
xi0 = xi[..., :mid]
xi2 = xi[..., mid:]

fig, axes = plt.subplots(2, 3, figsize=(7, 2.6),
                         gridspec_kw={'height_ratios': (3,1)})

# plot statistics
plot_model(axes[:,0], *vsf, 
           xlabel='$R_v$',
           ylabel='$n_v$ [$h^{3}$ Mpc$^{-3}$]')
plot_model(axes[:,1], *xi0,
           xlabel='s',
           ylabel=r'$\xi_0$')
plot_model(axes[:,2], *xi2,
           xlabel='s',
           ylabel=r'$\xi_2$')

axes[0,0].legend(loc='upper right', fontsize=fs * 3/4)

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.savefig("Figure2.pdf", bbox_inches='tight')
