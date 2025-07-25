import numpy as np
import matplotlib.pyplot as plt

def plot_DR1_like_comparison(fn=None):
    
    # load data
    DR1_like = np.load("power_LRG_A0_DR1_like.npy")
    DR1 = np.load("power_LRG_A0_DR1.npy")
    V1 = np.load("power_LRG_A0_V1.npy")
    
    # load Y1 error
    DR1_like_std = np.load("std_LRG_A0_DR1_like.npy")
    
    # plot
    labels = ['$P_\mathrm{HOD, DR1}$', '$P_\mathrm{HOD, V1}$', '$P_\mathrm{DR1}$']
    for ell in range(2): 
        plt.errorbar(DR1_like[0], DR1_like[0] * DR1_like[ell+1], yerr=DR1_like[0] * DR1_like_std[ell], 
                     color='k', label='' if ell>0 else labels[0])
        plt.plot(V1[0], V1[0] * V1[ell+1],
                 color='k', ls=':', label='' if ell>0 else labels[1])
        plt.plot(DR1[0], DR1[0] * DR1[ell+1], 
                 color='k', ls='--', label='' if ell>0 else labels[2])
    
    plt.xlim(0.02, 0.2)
    plt.xlabel('$k$ [h/Mpc]', fontsize=12)
    plt.ylabel('$kP(k)$ [(Mpc/h)$^2$]', fontsize=12)
    plt.grid()
    plt.legend()
    
    if fn is not None: plt.savefig(fn)

plot_DR1_like_comparison(fn="Figure1.pdf")