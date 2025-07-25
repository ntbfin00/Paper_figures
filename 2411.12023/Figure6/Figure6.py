import numpy as np
import matplotlib.pyplot as plt
from cosmoprimo.fiducial import DESI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DESIColors, DESIEdgeColors

# define fiducial cosmology
cosmo = DESI()

# colour settings for errorbars
def set_errorbar_color(handle, facecolor, edgecolor):
    handle.lines[0].set_color(edgecolor) # marker color
    handle.lines[0].set_markeredgecolor(edgecolor) # marker edge color
    handle.lines[0].set_markerfacecolor(facecolor)  # marker face color
    if len(handle.lines[1])>0:
        handle.lines[2][0].set_color(edgecolor) # errorbar color
        for line in handle.lines[1]: line.set_color(edgecolor) # cap color
            
def plot_tracers(fn=None):
    tracer_bins = ['BGS_0.1-0.4', 'LRG_0.4-0.6', 'LRG_0.6-0.8', 
                   'LRG_0.8-1.1', 'ELG_1.1-1.6', 'QSO_0.8-2.1']
    zeff = [0.3, 0.51, 0.71, 0.93, 1.32, 1.49]
    params = ['omega_cdm', 'h', 'logA', 'n_s']
    fid = {'omega_cdm': cosmo.Omega0_cdm * cosmo.h**2, 'h': cosmo.h, 
           'logA': np.log(10**10 * cosmo.A_s), 'n_s': cosmo.n_s}
    latex = {'omega_cdm': '$\\omega_{cdm}$', 'h': '$h$', 
             'logA': '$\\ln(10^{10} A_{s})$', 'n_s': '$n_{s}$'}
    edgecolors = [DESIEdgeColors[tracer] for tracer in ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']]
    facecolors = [DESIColors[tracer] for tracer in ['BGS', 'LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO']]

    # load mean and standard deviation of chains
    DR1_mean = np.load("chain_means_stds_DR1.npy")
    comb_mean = np.load("chain_means_stds_DR1+HOD.npy")

    # plot
    fig, ax = plt.subplots(len(params), 1, sharex=True)
    ax[0].set_xlim(0,1.6)
    offset = 0.02
    for (t,tracer) in enumerate(tracer_bins):
        for (p,param) in enumerate(params):
            if t<1: 
                ax[p].set_ylabel(latex[param], fontsize=12)
                fid_round = round(fid[param],2)
                ymax, ymin = (fid[param] + 1.7 * np.array([-DR1_mean[t,p,1], DR1_mean[t,p,1]]))
                ax[p].set_ylim(ymin, ymax)
                tick_step = round((ymin-ymax)/4,2)
                ax[p].set_yticks(tick_step*np.array([-1,0,1]) + fid_round) 
            ax[p].errorbar(zeff[t]-offset, DR1_mean[t,p,0], yerr=DR1_mean[t,p,1], ls='', marker='o', markersize=5, capsize=2,
                           color='darkgrey' if t<1 else facecolors[t], mec='dimgrey' if t<1 else edgecolors[t], label='DR1')
            ax[p].errorbar(zeff[t]+offset, comb_mean[t,p,0], yerr=comb_mean[t,p,1], ls='', marker='v', markersize=5, capsize=2,
                           color='darkgrey' if t<1 else facecolors[t], mec='dimgrey' if t<1 else edgecolors[t], label='DR1+HOD')
            ax[p].hlines(fid[param], 0, 1.6, color='k', ls='--', alpha=0.2, lw=1)
            
            # set figure legend
            if (t+p)==0: fig.legend(bbox_to_anchor=(0., 0.47, 1., 0.47), loc='upper center', ncols=3, 
                                    frameon=False, borderaxespad=0., fontsize=10)
            
            # set errorbar colors
            h = ax[p].get_legend_handles_labels()[0]
            set_errorbar_color(h[0], facecolors[t], edgecolors[t])
            set_errorbar_color(h[1], facecolors[t], edgecolors[t])

    ax[-1].set_xlabel('Redshift $z$')
    if fn is not None: plt.savefig(fn, bbox_inches="tight")

plot_tracers(fn="Figure6.pdf")