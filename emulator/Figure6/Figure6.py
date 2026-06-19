import numpy as np
from getdist import MCSamples, plots 
import matplotlib.pyplot as plt

# load chains
# chain_labels = ['prior', 'c000_vsf', 'c000_vg_ccf', 'c000_combined']
chain_labels = ['vsf', 'vg_ccf', 'combined']

chains = []
for label in chain_labels:
    samples = np.load(f"chain_c000_{label}.npy", allow_pickle=True)[()]
    # samples = np.load(f"chain_{label}.npy", allow_pickle=True)[()]
    chain = MCSamples(
                      samples=samples['samples'],
                      weights=samples['weights'],
                      names=samples['names'],
                      labels=[samples['labels'][name][1:-1] for name in samples['names']],
                      loglikes=samples['log_posterior'])
    chain.removeBurn(0.3)
    chains.append(chain)

# plot chains
# params = ['omega_cdm', 'sigma8_m']
params = ['omega_cdm', 'sigma8_m', 'n_s']
# params = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
# params = samples['names']
g = plots.get_subplot_plotter()
g.triangle_plot(chains, params=params, 
                filled=[True, True, False], 
                contour_lws=[2,2,2],
                contour_ls=['-','-','--','-'],
                contour_colors=['C0', 'C1', 'k', 'k'],
                markers=samples['markers'],
                # legend_labels = ['Prior', '$n_v$', r'$\xi_{vg}$'],
                legend_labels = ['VSF', 'CCF', 'Combined'],
                # shaded=[True, False, False]
                )
g.export("Figure6.pdf")
plt.show()
