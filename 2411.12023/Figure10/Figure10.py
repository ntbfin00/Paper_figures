import numpy as np
import matplotlib.pyplot as plt
from getdist import plots
from itertools import permutations
import scipy.stats as stats
from matplotlib import lines, patches
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import MCSamplesCompressed, DESIColors, DESIEdgeColors

def plot_method_comparison(fn=None):

    params = ['omega_cdm', 'h', 'logA', 'n_s']
    tracers = ['LRG', 'ELG']
    
    # set line defaults
    line1_kwargs = {'ls': '-', 'color': 'dimgrey', 'label': 'V1+HOD (General)'}
    line2_kwargs = {'ls': '-', 'color': 'k', 'label': 'V1+HOD (Restricted)'}

    # load mean and standard deviations from chain
    mean_std_nosys = np.load("chain_means_stds_nosys.npy")

    # load posteriors
    xlim = np.zeros((len(params),2))
    for (z, tracer) in enumerate(tracers):
        # load density contour values
        density_gen = MCSamplesCompressed.load(f"contour_{tracer}_general.npy")
        density_res = MCSamplesCompressed.load(f"contour_{tracer}_restricted.npy")
        
        # plot posteriors
        g = plots.get_subplot_plotter(subplot_size_ratio=0.6, width_inch=10)
        g.settings.line_labels = False
        g.settings.tight_layout = False
        g.settings.scaling_factor = 8/len(params)
        
        g.plots_1d([density_gen, density_res], 
                   params, nx=len(params), lws=1.5,
                   ls=[line1_kwargs['ls'], line2_kwargs['ls']], 
                   colors=[line1_kwargs['color'], line2_kwargs['color']])

        if z<1: 
            line1 = lines.Line2D([0,1],[1,1], **line1_kwargs)
            line2 = lines.Line2D([0,1],[1,1], **line2_kwargs)
            fill = patches.Patch(facecolor='lightgrey', edgecolor='dimgrey', 
                                 hatch='////', label='V1+HOD (Parameter level)')
            g.add_legend([], legend_loc='center', figure=True, legend_ncol=3, 
                         frameon=False, handles=[line1, line2, fill]);

        # Parameter level HOD contribution
        ML_V1_indiv = np.load(f'HODsys_MLV1_{tracer}.npy')

        # offsets to centre contours 
        for (p, param) in enumerate(params):
            ax = g.get_axes(ax=p)
            if z<len(tracers)-1: 
                ax.tick_params(bottom=False, labelbottom=False)
                ax.set_xlabel(None)
            if z<1: 
                xlim[p] = ax.get_xlim()
                xlim[p] = 0.5 * (xlim[p].sum() + np.diff(xlim[p])*[-1,1])
            ax.set_xlim(*xlim[p])

            # plot parameter-level gaussian contour
            x = np.arange(*(xlim[p] + [-0.1,0.1]*abs(xlim[p])), np.diff(ax.get_xlim())[0]/100)  # step for Gaussian PDF
            shifts = []
            for i in range(25):
                shifts.append(np.array([(m1-m2) for (m1,m2) in permutations(ML_V1_indiv[p,:,i],2)]))  # percent of fiducial
            shifts = np.concatenate(shifts)
            HOD_std = shifts.std()
            # combine HOD, PWE and external (imaging and fibre assignment) systematics
            comb_std = np.sqrt(mean_std_nosys[z,p,1]**2 + HOD_std**2) 
            pdf = stats.norm.pdf(x, mean_std_nosys[z,p,0], comb_std)
            # plot normalised gaussian contour
            ax.fill_between(x,  pdf * comb_std*np.sqrt(2*np.pi), label=tracer if p<1 else '',
                            color=DESIColors[tracer], edgecolor=DESIEdgeColors[tracer], lw=1.5)
            ax.legend(frameon=False, loc='upper left', fontsize=9);
            
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        if fn is not None: plt.savefig(fn.format(tracer), bbox_inches="tight")


# plot subplots 
plot_method_comparison(fn=f'Figure10_{{}}.pdf')