import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_sc

fs = 10  # fontsize

# load covariance mocks
cov_vsf = np.load("cov_vsf.npy")
cov_ccf = np.load("cov_vg_ccf.npy")

def get_quantiles(cov, Nsamp=100, Nsims=10_000):
    stds = cov.std(axis=0)
    means = cov.mean(axis=0)

    chi2_data = []
    N = cov.shape[0] - 1
    P = cov.shape[1]

    # compute chi-squared for covariance mocks using jacknife
    for i in range(0, len(cov)):
        y = cov[i]
        _cov = np.array(cov.tolist()[:i] + cov.tolist()[i+1:])
        covariance = np.cov(_cov.T)
        mu0 = np.mean(_cov, axis=0)
        inv = np.linalg.inv(covariance)
        inv *= (N - P - 2) / (N - 1)  # Hartlap correction
        _diff = y - mu0
        _chi2 = _diff.dot(inv.dot(_diff))
        chi2_data.append(_chi2)

    QQd = np.array(chi2_data)
    QQd /= P
    
    # bin theoretical chi-squared
    qedges = np.linspace(0., 1., Nsamp+1)
    qmid = 0.5*(qedges[:-1] + qedges[1:])
    QQt = chi2_sc.ppf(qmid, P)
    QQt /= P
    
    # bootstrap sample data chi-squared (Nsamp samples taken Nsims times)
    QQb = np.array([np.sort(np.random.choice(QQd, size=Nsamp, replace=True)) for _ in range(Nsims)])
            
    QQb_mean = np.mean(QQb, axis=0)
    QQb_std = np.std(QQb, ddof=1, axis=0)

    return QQt, QQb_mean, QQb_std

def plot_quantiles(cov, sigma=1, ax=None, label=None):

    QQt, QQb_mean, QQb_std = get_quantiles(cov)

    ax.plot(QQt, QQb_mean, color='C0', linestyle='-')

    ax.fill_between(QQt, 
                    QQb_mean - sigma * QQb_std, 
                    QQb_mean + sigma * QQb_std,
                        color='C0', alpha=0.5)
    ax.plot(QQt, QQt, 'k--')
    ax.set_xlabel(f"{label} theoretical quantiles", fontsize=fs)
    ax.set_ylabel(f"{label} data quantiles", fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs * 3/4)

    ax.set_xlim(QQt[0], QQt[-1])
    ax.text(0.05, 0.95, f'$p={cov.shape[1]:d}$', 
            horizontalalignment='left', verticalalignment='top', 
            transform=ax.transAxes, color='k', fontsize=10)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3))

plot_quantiles(cov_vsf, ax=ax1, label='$n_v$')
plot_quantiles(cov_ccf, ax=ax2, label=r'$\xi$')

plt.tight_layout()

plt.savefig("Figure5.pdf", bbox_inches='tight')
plt.show()
