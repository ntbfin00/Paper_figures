import numpy as np
from getdist import plots
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import MCSamplesCompressed, DESIColors, DESIEdgeColors

# Load MCMC contours
SN = MCSamplesCompressed.load("contour_SN.npy")
noSN = MCSamplesCompressed.load("contour_noSN.npy")

# Load MAP value for best-fit ELG model
MAP = np.load("MAP_mHMQ+cf+mNFW.py.npy")

plot_kwargs = {'filled': [True, False], 'contour_colors': ['darkgrey', DESIEdgeColors['ELG']], 'contour_lws': [2,2]}

# plot
g = plots.get_subplot_plotter()
g.settings.figure_legend_frame = False
g.settings.legend_fontsize = 17
g.settings.axes_labelsize = 20
g.settings.axes_fontsize = 15
g.triangle_plot([SN, noSN], ['omega_cdm', 'h', 'logA', 'n_s'], markers=MAP,
                legend_labels=["with shot-noise", "no shot-noise"], **plot_kwargs)
g.export("Figure8.pdf")