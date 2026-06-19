import numpy as np
import matplotlib.pyplot as plt

fs = 10  # fontsize

def triangle_plotter(values, names, diag=False, density=False, save_fn=None):
    n_dim = len(names) - 1
    
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(7, 7))

    for i in range(n_dim):
        for j in range(n_dim):
    
            ax = axes[i, j]
    
            if i < j:
                ax.axis("off")  # upper triangle
    
            elif (i == j):
                if diag:
                    ax.hist(values[:, i], bins=30, histtype="step", color='k')
                else:
                    ax.axis('off')
            else:
                if density:
                    hist = ax.hist2d(values[:, j], values[:, i], 
                                     rasterized=True, bins=50)[3]
                else:
                    vals = np.unique([values[:,j], values[:, i]], axis=1)
                    ax.scatter(*vals, s=1, alpha=0.8, c='k')
    
            # Labels only on outer plots
            if i == n_dim - 1:
                ax.set_xlabel(names[j], fontsize=fs)
                ax.tick_params(axis='x', which='major', 
                               labelsize=fs * 3/4, rotation=90)
            else:
                ax.set_xticklabels([])
                ax.set_xticks([])
    
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=fs)
                ax.tick_params(axis='y', which='major', labelsize=fs * 3/4)
            else:
                ax.set_yticklabels([])
                ax.set_yticks([])

            if not diag and (i==0):
                ax.remove()
                
    if density:
        valid_axes = [ax for ax in axes.ravel() if ax.has_data()]
        cbar = fig.colorbar(hist, ax=valid_axes, 
                            orientation='horizontal', location='top', fraction=0.05,
                            ticks=np.arange(0, 20, 2), shrink=0.5, aspect=15)#, fraction=0.3)
        cbar.set_label('Counts', size=fs)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.tick_params(labelsize=fs * 3/4)



    plt.subplots_adjust(wspace=0, hspace=0)
    if save_fn is not None: plt.savefig(save_fn, bbox_inches='tight')

param_names = [r'$\omega_\mathrm{b}$', r'$\omega_\mathrm{cdm}$', r'$\sigma_8$', '$n_s$', r'$n_\mathrm{run}$',
               r'$N_\mathrm{ur}$', '$w_0$', '$w_a$', r'$\log M_\mathrm{cut}$', r'$\log M_1$',
               r'$\log \sigma$', r'$\alpha$', r'$\kappa$', r'$\alpha_c$', r'$\alpha_s$',
               '$s$', r'$A_\mathrm{cen}$', r'$A_\mathrm{sat}$', r'$B_\mathrm{cen}$', r'$B_\mathrm{sat}$']

params = np.load("abacus_params.npy")

triangle_plotter(params[:,:9], param_names[:9],
                 save_fn="cosmo_params.pdf")

triangle_plotter(params[:,8:], param_names[8:], density=True,
                 save_fn="hod_params.pdf")
