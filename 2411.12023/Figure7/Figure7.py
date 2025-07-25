import numpy as np
from getdist import MCSamples, plots
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import MCSamplesCompressed, DESIColors, DESIEdgeColors

# Load MCMC contours
DR1 = MCSamplesCompressed.load("contour_DR1.npy")
comb = MCSamplesCompressed.load("contour_DR1+HOD.npy")

plot_kwargs = {'filled': [True, False], 'contour_colors': ['darkgrey', DESIEdgeColors['ELG']], 'contour_lws': [2,2]}

# plot
g = plots.get_subplot_plotter()
g.settings.figure_legend_frame = False
g.settings.legend_fontsize = 17
g.settings.axes_labelsize = 20
g.settings.axes_fontsize = 15
g.triangle_plot([DR1, comb], ['omega_cdm', 'h', 'logA', 'n_s'], 
                legend_labels=['DR1','DR1+HOD'], **plot_kwargs)
g.export("Figure7.pdf")