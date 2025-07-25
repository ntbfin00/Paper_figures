import numpy as np
import matplotlib.pyplot as plt

def cov2corr(cov):
    v = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(v, v)
    corr[~np.isfinite(corr)] = 0.0
    return corr
        
def plot_covariance(fn=None):

    fig, axes = plt.subplots(2,2, figsize=(8, 8), 
                             gridspec_kw={'width_ratios': [0.94,1]})
    for i in range(axes.size):
        if (i%2==0):
            col = 0
            tracer = 'LRG'
            cbar = False
        else:
            col = 1
            tracer = 'ELG'
            cbar = True
        if (i//2==0):
            row = 0
            HOD = ""
            title = True
            xticks = False
        else:
            row = 1
            HOD = "+HOD"
            title = False
            xticks = True

        # load covariance matrix and convert to correlation coefficient
        covariance = np.load(f"cov_{tracer}_DR1{HOD}.npy")
        corr = cov2corr(covariance)

        # plot correlation matrix
        ax = axes[row, col]
        ax.matshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        # add colorbar
        if cbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
            fig.colorbar(im, cax=cax, orientation='vertical', 
                         ticks=np.arange(-1, 1.5, 0.5))
        ticks = np.array(range(4)) * int(covariance.shape[0]/4) - 0.5
        # add xticks
        if xticks: ax.set_xticks(ticks[1::2], labels=['$P_0$', '$P_2$'], minor=True, fontsize=14)
        # add yticks
        if not cbar: ax.set_yticks(ticks[1::2], labels=['$P_0$', '$P_2$'], minor=True, fontsize=14)
        ax.set_xticks([ticks[2]], labels=[''], minor=False)
        ax.set_yticks([ticks[2]], labels=[''], minor=False)
        ax.xaxis.set_ticks_position('bottom')
        ax.grid(which='major')
        # add tracer title
        if title: ax.set_title(tracer, fontsize=15)

    plt.tight_layout()
    if fn is not None: plt.savefig(fn, bbox_inches="tight")

plot_covariance("Figure5.pdf")