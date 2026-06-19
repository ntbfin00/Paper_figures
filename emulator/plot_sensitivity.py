import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.collections import LineCollection
import acm.observables.emc as emc
import argparse
from acm.utils.paths import lookup_registry_path

parser = argparse.ArgumentParser(description='Plot emulator response to cosmology.')
# parser.add_argument('-s', '--statistic', type=str, default='VERSUSVoidSizeFunction')
# parser.add_argument('-p', '--param', type=str, default='omega_cdm')
parser.add_argument('--delta', type=float, default=3)
parser.add_argument('--n_steps', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='fig/',)
args = parser.parse_args()

fs = 10

# observable_name = args.statistic
# param = args.param
delta = args.delta
n_steps = args.n_steps
param_range = [1-delta/100, 1+delta/100, n_steps+1]

paths = lookup_registry_path('projects.yaml', 'emc')

# kwargs = {'paths': paths, 'numpy_output': True, 'squeeze_output': True, 'select_filters': {'cosmo_idx': 0, 'hod_idx': 0}}
# observable = getattr(emc, observable_name, None)(**kwargs)

# base_params.update({'w0_fld': -0.7, 'wa_fld': -0.5})

ylabel = {'vsf': r'$n_{\rm void}\,[h^3{\rm Mpc}^{-3}]$', 'xivg': rf'$\xi_{{}}(s)$', 'xivv': rf'$\xi_{{}}(s)$'}
xlabel = {'vsf': r'$R_{\rm void}\, [h^{-1}{\rm Mpc}]$', 'xivg': r'$s [h^{-1}{\rm Mpc}]$', 'xivv': r'$s [h^{-1}{\rm Mpc}]$'}

latex = {'omega_b': r'$\omega_{\rm b}$', 'omega_cdm': r'$\omega_{\rm cdm}$', 
         'sigma8_m': r'$\sigma_8$', 'n_s': r'$n_s$', 'w0_fld': r'$w_0$', 'wa_fld': r'$w_a$',
         'logM_cut': '$\log M_\mathrm{cut}$', 'logM_1': '$\log M_1$', 'sigma': '$\log \sigma$', 
         'alpha': '$\alpha$', 'kappa': '$\kappa$', 'alpha_c': '$\alpha_c$', 'alpha_s': '$\alpha_s$', 's': '$s$', 
         'A_cen': '$A_\mathrm{cen}$', 'A_sat': '$A_\mathrm{sat}$', 'B_cen': '$B_\mathrm{cen}$', 'B_sat': '$B_\mathrm{sat}$'}

def store_predictions(observable, param, param_range):
    pred = []
    base_params = dict(zip(observable.x_names, observable.x))
    for i in np.linspace(*param_range):
        params = base_params.copy()
        params.update({param: params[param] * i})
        pred.append(observable.get_model_prediction(params))
        
    return np.array(pred)

def plot_lines(x, predictions, delta, ax, err=None, **kwargs):
    pred = np.array(predictions)
    if err is not None:
        pred /= err
    lines = LineCollection(np.stack((np.broadcast_to(x, pred.shape), pred), axis=-1), 
                           array=np.linspace(-delta, delta, pred.shape[0]), **kwargs)
    ax.add_collection(lines)
    ax.autoscale()
    
    return lines


statistics = ['VERSUSVoidSizeFunction', 'VERSUSVoidGalaxyCorrelationFunctionMultipoles']
params = ['omega_cdm', 'sigma8_m', 'w0_fld']#, 'wa_fld']

fig, axes = plt.subplots(len(params), 3, figsize=(7, 5), sharey=True)

for (p,param) in enumerate(params):
    axes[p,0].set_ylabel(rf'$\Delta X$({latex[param]}) / $\sigma_\mathrm{{V1}}$')
    axes[p,0].set_yticks(range(-2,3))
    for (i,statistic) in enumerate(statistics):
        print(f"Plotting {statistic}...")
        observable = getattr(emc, statistic)(paths=paths, numpy_output=True, squeeze_output=True,
                                             select_filters={'cosmo_idx': 0, 'hod_idx': 43})
        pred = store_predictions(observable, param, param_range)
        idx_zero = (pred.shape[0] + 1) // 2
        err = np.sqrt(np.diag(observable.get_covariance_matrix()))
        try:
            x = observable.rv
            pred = [pred]
            # err = [err]
            err = [np.sqrt(observable.y / 2000**3)]
        except:
            x = observable.s
            mid = pred.shape[-1] // 2
            pred = [pred[:, :mid] , pred[:, mid:]]
            err = [err[:mid] , err[mid:]]
        xlims = x.min(), x.max()

        for j in range(i+1):
            # lines = plot_lines(x, pred[j], delta, axes[p, i+j], cmap="BrBG_r", linewidths=2)
            lines = plot_lines(x, pred[j] - pred[j][idx_zero], delta, axes[p, i+j], err=err[j], 
                               cmap="BrBG_r", linewidths=2)
            axes[p,i+j].set_xlim(*xlims)
            axes[p,i+j].set_ylim(-3, 3)
            axes[p,i+j].tick_params(axis='both', which='major', labelsize=fs * 3/4)
            if p < len(params)-1:
                axes[p,i+j].set_xticks([])
            else:
                axes[p,i+j].text(0.98, 0., ["$n_v$", r"$\xi_0$", r"$\xi_2$"][i+j],
                               transform=axes[p,i+j].transAxes,
                               fontsize=1.5*fs, ha="right", va="bottom")
            if i+j > 0: 
                axes[p,i+j].tick_params(labelleft=False)
                # axes[p,i+j].set_yticks([])

    # axes[p,0].set_ylabel(r'$n_v\, [h^3{\rm Mpc}^{-3}]$', fontsize=fs)
    # axes[p,1].set_ylabel(r'$\xi_0$', fontsize=fs)
    # axes[p,2].set_ylabel(r'$\xi_2$', fontsize=fs)

axes[-1,0].set_xlabel(r'$R_v$ [$h^{-1}{\rm Mpc}$]', fontsize=fs)
axes[-1,1].set_xlabel(r'$s$ [$h^{-1}{\rm Mpc}$]', fontsize=fs)
axes[-1,2].set_xlabel(r'$s$ [$h^{-1}{\rm Mpc}$]', fontsize=fs)

# fig.colorbar(lines, ax=axes.ravel().tolist(), label=r"$\Delta \Omega$ (%)",
#              orientation='horizontal', location='top', fraction=0.05)
fig.subplots_adjust(right=0.88, wspace=0, hspace=0)

cax = fig.add_axes([0.90, 0.15, 0.025, 0.7])

cbar = fig.colorbar(
    lines,
    cax=cax,
    orientation='vertical'
)

cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.yaxis.set_label_position('right')
cbar.ax.tick_params(labelsize=fs * 3/4)
cbar.set_label(r"$\Delta \Omega$ (%)", fontsize=fs)

# plt.tight_layout()
# plt.subplots_adjust(wspace=0., hspace=0)
# plt.savefig(args.save_dir + f'predictions_{observable.stat_name}.png')
plt.savefig(args.save_dir + f'predictions_{observable.stat_name}.pdf', bbox_inches='tight')