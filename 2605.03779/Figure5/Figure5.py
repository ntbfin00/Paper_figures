import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from healpy.newvisufunc import projview

colors = plt.get_cmap("Paired").colors
fs = 10  # fontsize
ms = 3  # markersize
lw = 1.5  # linewidth

def bin_healpix(random_fn="CMASS_randoms.npy", nside=256, save_output=True):

    pos = np.load(random_fn)
    x_obs = pos[:,0]
    y_obs = pos[:,1]
    z_obs = pos[:,2]

    # spherical coords
    r = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
    ra = np.degrees(np.arctan2(y_obs, x_obs)) % 360
    dec = np.degrees(np.arcsin(z_obs / r))
    
    # Convert to radians
    theta = np.radians(90.0 - dec)   # colatitude
    phi   = np.radians(ra % 360.0)   # longitude

    # Pixel indices
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    npix = hp.nside2npix(nside)

    # Count number of randoms per pixel
    counts = np.bincount(pix, minlength=npix)

    if save_output: 
        np.save("healpix_counts.npy", counts)

    return counts

# optionally determine healpix random counts 
# counts = bin_healpix()

# generate healpix mask
counts = np.load("healpix_counts.npy")
mask = hp.ma(counts, badval=0)

# load VSFs
box_vsf = np.load("box_vsf.npy")
cutsky_vsf = np.load("CMASS_vsf.npy")

# create figure
fig = plt.figure(figsize=(7, 2))

# plot footprint
projview(
    mask,
    sub=(1,2,1),
    dpi=5000,
    badcolor='white',
    graticule=True,
    graticule_labels=True,
    longitude_grid_spacing=20,
    latitude_grid_spacing=20,
    xlabel="RA [degrees]",
    ylabel="DEC [degrees]",
    projection_type="cart",
    lonra=(-50, 50),
    latra=(-20,40),
    phi_convention='symmetrical',
    cbar=False,
    fontsize={'xlabel': fs, 'ylabel': fs,
              'xtick_label': fs * 3/4, 'ytick_label': fs * 3/4}
);

# plot VSF comparison
ax = fig.add_subplot(1, 2, 2, box_aspect=0.8)
ax.errorbar(*box_vsf, label='Box ($z=0.5$)', c=colors[1], lw=lw)
ax.errorbar(*cutsky_vsf, label='CMASS ($0.4 < z < 0.6$)', 
            c='k', lw=lw, ls='--', zorder=100)
ax.set_ylabel(r'$d n_v / d \ln R_v$ [$h^4{{\rm Mpc}}^{{-4}}$]', fontsize=fs)
ax.set_xlabel('$R_v$ [$h^{-1}$Mpc]', fontsize=fs)
ax.tick_params(axis='both', which='major', labelsize=fs * 3/4)
ax.tick_params(axis='both', which='minor', labelsize=fs * 3/4)
ax.set_ylim(25, 61)
ax.set_ylim(0, 6e-6)
ax.legend(fontsize=fs * 3/4)

plt.subplots_adjust(wspace=0)
plt.savefig("Figure5.pdf", bbox_inches='tight')
plt.show()
