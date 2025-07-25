import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DESIColors, DESIEdgeColors

def plot_pks(z, ax=None, ylabel=True, fn=None):
    k = np.load("k.npy")
    pks = np.load(f"pk_{z}.npy")
    if z=="LRG": bf_HOD = "A0"; bf_HOD_indx = 0
    if z=="ELG": bf_HOD = "mHMQ+cf+mNFW"; bf_HOD_indx = 19

    if ax is None: fig, ax = plt.subplots(1)
    for pk in pks:
        ax.plot(k, k*pk[0], color=DESIEdgeColors[z], lw=1, alpha=0.6)
        ax.plot(k, k*pk[1], color=DESIEdgeColors[z], lw=1, alpha=0.6)
    ax.plot(k, k*pks[bf_HOD_indx,0], lw=2, color='k', label=bf_HOD)
    ax.plot(k, k*pks[bf_HOD_indx,1], lw=2, color='k')
    if ylabel: ax.set_ylabel('$kP(k)$ [(Mpc/h)$^3$]', fontsize=12)
    ax.set_xlabel('$k$ [h/Mpc]', fontsize=12)
    ax.set_xlim(0.02,0.2)
    ax.legend(fontsize=10, frameon=False)
    if fn is not None: plt.savefig(fn, bbox_inches="tight")

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,3.5))
plot_pks('LRG', ax=ax1)
plot_pks('ELG', ax=ax2, ylabel=False)
plt.tight_layout()
plt.savefig('Figure9.pdf', bbox_inches="tight")