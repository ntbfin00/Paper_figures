import numpy as np
import matplotlib.pyplot as plt
from cosmoprimo.fiducial import DESI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DESIColors, DESIEdgeColors

# define fiducial cosmology
cosmo = DESI()

# set models based on redshift or tracer type
def set_HOD_labels(tracer):
    models = ['A0', 'A1', 'A2', 'A3', 
              'B0', 'B1', 'B2', 'B3', 
              'HMQ_1', 'HMQ_2', 'HMQ_3', 'HMQ_4', 'HMQ_5', 'HMQ_6',
              'GHOD', 
              'LNHOD', 'LNHOD-1h', 'LNHOD+cf', 'LNHOD2', 'LNHOD2+cf', 
              'SFHOD', 'SFHOD+cf', 
              'HMQ', 
              'mHMQ', 'mHMQ+cf', 'mHMQ+cf+C', 'mHMQ+cf+Env', 'mHMQ+cf+mNFW', 'mHMQ+cf+Sh',
              'BGS_0', 'BGS_1', 'BGS_2', 'BGS_3', 'BGS_4', 'BGS_5', 'BGS_6', 'BGS_7', 'BGS_8', 'BGS_9', 'BGS_10',
              'QSO_0', 'QSO_1', 'QSO_2', 'QSO_3']
    
    # convert to latex
    models[8:14] = [f'{m[:-2]}${m[-2:]}^{{(3\sigma)}}$' for m in models[8:14]]
    models[29:] = [f'{m[:3]}$_{{{m[4:]}}}$' for m in models[29:]]
    
    if tracer == 'LRG': return models[:8]
    if tracer == 'ELG': return models[8:29]
    if tracer == 'BGS': return models[29:40]
    if tracer == 'QSO': return models[40:]

# colour settings for errorbars
def set_errorbar_color(handle, facecolor, edgecolor):
    handle.lines[0].set_color(edgecolor) # marker color
    handle.lines[0].set_markeredgecolor(edgecolor) # marker edge color
    handle.lines[0].set_markerfacecolor(facecolor)  # marker face color
    if len(handle.lines[1])>0:
        handle.lines[2][0].set_color(edgecolor) # errorbar color
        for line in handle.lines[1]: line.set_color(edgecolor) # cap color

# plot values for all HOD models
def plot_HOD_shifts(fn=None):

    params = ['omega_cdm', 'h', 'logA', 'n_s']
    fid = {'omega_cdm': cosmo.Omega0_cdm * cosmo.h**2, 'h': cosmo.h, 
           'logA': np.log(10**10 * cosmo.A_s), 'n_s': cosmo.n_s}
    latex = {'omega_cdm': '$\\omega_{cdm}$', 'h': '$h$', 
             'logA': '$\\ln(10^{10} A_{s})$', 'n_s': '$n_{s}$'}
    tracers = ['BGS','LRG','ELG','QSO']
    labels = ['$\sigma^\mathrm{DR1}_\mathrm{stat}$', 'MAP V1 (uninformative)', 'MAP DR1 (physical)']
    N_HOD = [len(set_HOD_labels(tracer)) for tracer in tracers]
    
    # set y-limit based on BGS
    ylim = [1.85 * abs(err - fid[params[p]]).max() for (p,err) in enumerate(np.load(f'Y1err_BGS.npy'))]

    fig, axes = plt.subplots(len(params), len(tracers), 
                             gridspec_kw={'width_ratios': N_HOD},
                             figsize=(4*len(tracers),1.5*len(params)))

    for (t, tracer) in enumerate(tracers):
        # load Y1 error band
        Y1err = np.load(f'Y1err_{tracer}.npy')
        # load V1 fits for individual realisations
        ML_V1 = np.load(f'MLV1_{tracer}.npy')
        # load mean Y1 fits
        if tracer=='BGS':
            MAP_Y1 = np.full((len(params), N_HOD[t]), None)  # no Y1-like data for BGS
        else:
            MAP_Y1 = np.load(f'MAPY1_{tracer}.npy')

        for (p, param) in enumerate(params):
            # plot Y1 error band
            axes[p,t].fill_between([-0.5, N_HOD[t]], *np.full((2,2), Y1err[p]).transpose(), 
                                   color='darkgrey', alpha=0.5, label=labels[0])
            # plot fiducial values
            axes[p,t].hlines(fid[param], -0.5, N_HOD[t]-0.5, ls='--', color='k', lw=1, alpha=0.3)
            
            # marker and errorbar settings
            xticks = np.arange(N_HOD[t])
            offset = 0 if tracer=='BGS' else 0.15
            yerr = None if tracer=='BGS' else ML_V1[p].std(axis=-1)
            mean = ML_V1[p,:,0] if tracer=='BGS' else ML_V1[p].mean(axis=-1)
            
            # plot ML V1 means with error bars
            axes[p,t].errorbar(xticks-offset, mean, yerr=yerr, 
                               markersize=5.5, capsize=2, linestyle='', marker='v', mew=1,
                               mfc='darkgrey', color='dimgrey', label=labels[1])
            # plot MAP Y1
            axes[p,t].errorbar(xticks+offset, MAP_Y1[p], yerr=None, 
                               markersize=5, linestyle='', marker='x', 
                               mfc='darkgrey', mec='dimgrey', label=labels[2])
            
            # set figure legend
            if (t+p)==0: fig.legend(bbox_to_anchor=(0., 0.49, 1., 0.47), loc='upper center', ncols=3, 
                                    frameon=False, borderaxespad=0., fontsize=14)
            
            # set errorbar colors
            h = axes[p,t].get_legend_handles_labels()[0]
            h[0].set_color(DESIColors[tracer])  # set Y1 band color
            h[0].set_edgecolor(DESIEdgeColors[tracer])  # set Y1 band color
            if p<1:  # add tracer as subplot legend
                axes[0,t].legend(handles=[h[0]], labels=[tracer], 
                                 loc='upper left', fontsize=9, frameon=False) 
            h[0].set_edgecolor(None)
            h[0].set_alpha(0.2)
            set_errorbar_color(h[1], DESIColors[tracer], DESIEdgeColors[tracer])  # set ML V1 colors
            set_errorbar_color(h[2], DESIColors[tracer], DESIEdgeColors[tracer])  # set MAP Y1 colors
            
            # subplot settings
            axes[p,0].set_ylabel(latex[param], fontsize=15)
            ymin, ymax = ylim[p]*np.array([-1,1]) + fid[param]
            fid_round = round(fid[param],2)
            tick_step = round((ymin-ymax)/4,2)
            axes[p,0].set_yticks(tick_step*np.array([-1,0,1]) + fid_round)            
            axes[p,t].set_ylim(ymin, ymax)
            axes[p,t].set_xlim(-0.5, N_HOD[t]-0.5)
            if p<(len(params)-1): axes[p,t].tick_params(bottom=False, labelbottom=False)
            if t>0:  axes[p,t].tick_params(left=False, labelleft=False)
        axes[-1,t].set_xticks(range(N_HOD[t]), labels=set_HOD_labels(tracer), fontsize=10)
        plt.setp(axes[-1,t].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    fig.subplots_adjust(hspace=0, wspace=0.05)
    if fn is not None: plt.savefig(fn, bbox_inches="tight")


# plot figure
plot_HOD_shifts(fn='Figure2.pdf')