import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DESIColors, DESIEdgeColors

def get_diag(k, std, axes, label='', fill=False, **kwargs):  
    # std = np.sqrt(np.diag(cov))
    # std = std.reshape(k.shape)

    plot = [ax.errorbar for ax in axes] if fill is False else [ax.fill_between for ax in axes]
    
    if type(fill) is bool:
        plot[0](k, k*-std[0] if fill else np.zeros_like(k), k*std[0], label=label, **kwargs)
        plot[1](k, k*-std[1] if fill else np.zeros_like(k), k*std[1], **kwargs)
    else:
        plot[0](k, k*-std[0], k*-fill[0], label=label, **kwargs)
        plot[0](k, k*fill[0], k*std[0], **kwargs)
        plot[1](k, k*-std[1], k*-fill[1], **kwargs)
        plot[1](k, k*fill[1], k*std[1], **kwargs)
            #     std_fill = np.sqrt(np.diag(fill))
    #     std_fill = std_fill.reshape(k.shape)
    #     plot[0](k[0], k[0]*-std[0], k[0]*-std_fill[0], label=label, **kwargs)
    #     plot[0](k[0], k[0]*std_fill[0], k[0]*std[0], **kwargs)
    #     plot[1](k[1], k[1]*-std[1], k[1]*-std_fill[1], **kwargs)
    #     plot[1](k[1], k[1]*std_fill[1], k[1]*std[1], **kwargs)

def plot_diagonal(fn=None):
    
    fig, axes = plt.subplots(2,2, figsize=(10,4), sharex=True)
    # k = np.array(get_wmatrix('LRG').k)
        
    # cov_fn = 'C_HOD{}{}.npy'.format('_window' if window else '', '_rotated' if rotated else '')
    # dataset = 'DR1' if window else 'V1'
    for (t,tracer) in enumerate(['LRG','ELG']):
        # load square-root of HOD covariance diagonal 
        HOD = np.load(f"std_{tracer}_HOD.npy")
        k = HOD[0]
        std_HOD = HOD[1:]
        
        # load square-root of DR1 covariance diagonal 
        DR1 = np.load(f"std_{tracer}_DR1.npy")
        std_DR1 = DR1[1:]

        get_diag(k, std_DR1, axes[:,t], label='DR1 error', color='darkgrey', fill=True)
        get_diag(k, std_HOD, axes[:,t], color='k', label='HOD error', capsize=2, elinewidth=1.5)
        get_diag(k, np.sqrt(std_DR1**2 + std_HOD**2), axes[:,t], fill=std_DR1, label='DR1+HOD error',
                 color=DESIColors[tracer], edgecolor=DESIEdgeColors[tracer])
        
        # set tracer legend
        h = axes[0,t].get_legend_handles_labels()[0]
        axes[0,t].legend(handles=[h[1]], labels=[tracer], loc='upper right', fontsize=11, frameon=False)

    axes[0,0].set_ylabel('$kP_0(k)$ [(Mpc/h)$^2$]', fontsize=12)
    axes[1,0].set_ylabel('$kP_2(k)$ [(Mpc/h)$^2$]', fontsize=12)
    axes[1,0].set_xlabel('$k$ [h/Mpc]', fontsize=12)
    axes[1,0].set_xlim((0.02,0.2))
    axes[1,1].set_xlabel('$k$ [h/Mpc]', fontsize=12)
    axes[1,1].set_xlim((0.02,0.2))
    
    # set figure legend
    # h = axes[0,0].get_legend_handles_labels()[0]
    h[1].set_color('lightgrey')
    h[1].set_edgecolor('black')
    h[1].set_alpha(0.6)
    fig.legend(handles=[h[2],h[0],h[1]], bbox_to_anchor=(0., 0.56, 1., 0.55), 
               loc='upper center', ncols=3, frameon=False, fontsize=12)
    h[1].set_color(DESIColors['ELG'])
    h[1].set_edgecolor(DESIEdgeColors['ELG'])
    h[1].set_alpha(1)
    
    plt.tight_layout(pad=0.)
    fig.subplots_adjust(hspace=0.)
    
    if fn is not None: plt.savefig(fn, bbox_inches="tight")

plot_diagonal(fn="Figure4.pdf")