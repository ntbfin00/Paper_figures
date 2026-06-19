import numpy as np
from getdist import MCSamples, plots 
import matplotlib.pyplot as plt

Ncosmos = 5

chains = []
markers = []
for cosmo in range(5):
    samples = np.load(f"chain_c{cosmo:03d}_combined.npy", allow_pickle=True)[()]
    chain = MCSamples(
                      samples=samples['samples'],
                      weights=samples['weights'],
                      names=samples['names'],
                      labels=[samples['labels'][name][1:-1] for name in samples['names']],
                      loglikes=samples['log_posterior'])
    chain.removeBurn(0.3)
    chains.append(chain)
    markers.append(samples['markers'])

# plot chains
params = ['omega_cdm', 'sigma8_m', 'w0_fld', 'wa_fld']
# params = samples['names']
colors = [f'C{i}' for i in range(Ncosmos)]
g = plots.get_subplot_plotter()
chains += chains
g.triangle_plot(chains, params=params, 
                filled=[True for i in range(Ncosmos)]
                      +[False for i in range(Ncosmos)], 
                contour_lws=[2 for i in range(2*Ncosmos)],
                contour_ls=['-' for i in range(2*Ncosmos)],
                contour_colors=colors+colors,
                alphas=0.5,
                legend_labels = [f'c{cosmo:03d}' for cosmo in range(Ncosmos)],
                )

# add markers
n = len(params)
axes = [i * n + j for i in range(n) for j in range(i)]
axes_params = [
             (params[col], params[row])
             for row in range(1, n)
             for col in range(row)
         ]

for i in range(Ncosmos):
    marker = markers[i]
    for (j,p) in enumerate(axes):
        ax = g.get_axes(p)
        ax.scatter(marker[axes_params[j][0]],
                   marker[axes_params[j][1]],
                   color=colors[i], edgecolors='black', marker='X')

g.export("Figure7.pdf")
plt.show()
