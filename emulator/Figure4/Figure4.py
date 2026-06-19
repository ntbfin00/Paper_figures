import numpy as np
from getdist import MCSamples, plots 

# load chains
chain_labels = ['gauss_like', 'poisson_like', 'gauss_like_poisson_cov'] 
# chain_labels = ['gauss_like', 'poisson_like', 'gauss_like_poisson_cov', 'gauss_like_poisson_cov_w_norm']

chains = []
for label in chain_labels:
    samples = np.load(f"chain_{label}.npy", allow_pickle=True)[()]
    chain = MCSamples(
                      samples=samples['samples'],
                      weights=samples['weights'],
                      names=samples['names'],
                      labels=[samples['labels'][name][1:-1] for name in samples['names']],
                      loglikes=samples['log_posterior'])
    chain.removeBurn(0.3)
    chains.append(chain)

# plot chains
params = ['omega_cdm', 'sigma8_m', 'n_s']
g = plots.get_subplot_plotter()
g.triangle_plot(chains, params=params, 
                filled=[True, True, False, False], 
                contour_lws=[2,2,2,2],
                contour_ls=['-','-','--','-'],
                contour_colors=['C0', 'C1', 'k', 'k'],
                markers=samples['markers'],
                legend_labels = ['Gaussian likelihood', 'Poisson likelihood', "Gaussian likelihood + 'Poisson' covariance", "... + normalisation term"])
g.export("Figure4.pdf")

import matplotlib.pyplot as plt
plt.show()
