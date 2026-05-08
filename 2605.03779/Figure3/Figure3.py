import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.patches import Circle 
import matplotlib.colors as colors


fs = 10  # fontsize
ms = 0.01  # markersize
lw = 1.5  # linewidth

boxlims = (-1000, 1000)
ncells = 500
axes_labels = ['X','Y','Z']

def load_slice(data_fn="data_positions.npy", 
               voids_fn="versus_voids.npy", 
               membership_fn="versus_membership.npy",
               slice_axis = 'Z',
               slice_range = (30, 80),
               inset_void_range = (550, 700),
               save_output=True):

    # load data
    data_positions = np.load(data_fn)
    voids      = np.load(voids_fn)
    ids        = voids[:,0]
    positions  = voids[:,1:4]
    radii      = voids[:,4]
    membership = np.load(membership_fn)

    # slice parameters
    axis = axes_labels.index(slice_axis.upper())
    axes = [0, 1, 2]
    axes.pop(axis)

    # filter voids
    mask_sp = ((positions[:,axis] >= slice_range[0]) 
            & (positions[:,axis] <= slice_range[1]))
    # void_pos = positions[mask_sp]
    # void_rad = radii[mask_sp]
    voids_slice = voids[mask_sp]
    voids_slice = np.delete(voids_slice, 0, axis=1)
    voids_slice = np.delete(voids_slice, axis, axis=1)

    # filter cells around chosen void
    mask_void = ((voids_slice[:,0] >= inset_void_range[0]) 
               & (voids_slice[:,0] <  inset_void_range[1])
               & (voids_slice[:,1] >= inset_void_range[0]) 
               & (voids_slice[:,1] <  inset_void_range[1]))
    inset_indx = np.argmax(mask_void)

    # filter galaxies 
    mask_xy = ((data_positions[:,axis] >= slice_range[0]) 
            & (data_positions[:,axis] <= slice_range[1]))
    pos_slice = data_positions[mask_xy]

    mesh_slice = np.histogram2d(pos_slice[:,axes[0]], pos_slice[:,axes[1]], bins=ncells)[0]

    # filter membership cells
    lim_z = (np.array(slice_range) + 1000) / 4
    membership_slice = membership.take(indices=range(int(lim_z[0]), int(lim_z[1])), axis=axis)

    if save_output:
        np.save("mesh_slice.npy", mesh_slice)
        np.save("voids_slice.npy", voids_slice)
        np.save("membership_slice.npy", membership_slice)

    return mesh_slice, voids_slice, membership_slice, inset_indx


def plot_slice(mesh_slice, voids_slice, membership_slice,
               slice_axis = 'Z',
               slice_range = (30, 80),
               inset_indx=23,
               inset_pad=80,
               cmap = 'viridis',
               inset_color = 'lightgrey',
               save_fn=None):
               
    # create axes with inset
    fig, ax = plt.subplots(figsize=(10/3, 10/3))
    ax_zoom = zoomed_inset_axes(ax, zoom=5, loc='lower right')
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor(inset_color)

    # plot galaxies
    ax.imshow(mesh_slice.T, origin='lower', extent=(*boxlims, *boxlims),
              cmap=cmap, norm=colors.PowerNorm(0.3), rasterized=True)
    for (i, void) in enumerate(voids_slice):
        circ = Circle((void[0], void[1]), void[2], fill=False, edgecolor='goldenrod', 
                      label=None if i>0 else 'Void')
        ax.add_patch(circ)

    ax.set_xlabel(axes_labels[0] + r' $[h^{-1}{\rm Mpc}]$', fontsize=fs)
    ax.set_ylabel(axes_labels[1] + r' $[h^{-1}{\rm Mpc}]$', fontsize=fs)
    ax.set_xlim(boxlims)
    ax.set_ylim(boxlims)
    ax.set_title((rf'{slice_range[0]} $\leq$' 
                + rf'{slice_axis.upper()} $[h^{{-1}}{{\rm Mpc}}]$' 
                + rf'$\leq$ {slice_range[1]}'),
                fontsize=fs)
    ax.set_aspect('equal')
    ticks = np.linspace(-1000, 1000, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', which='major', labelsize=fs * 3/4)

    # filter membership cells in inset
    inset_xy = np.array([voids_slice[inset_indx, 0], voids_slice[inset_indx,1]]) - inset_pad
    lim_xy = (inset_xy + 1000) / 4
    axis = axes_labels.index(slice_axis.upper())
    axes = [0, 1, 2]
    axes.pop(axis)
    for i in range(len(axes)):
        membership_slice = membership_slice.take(indices=range(int(lim_xy[i]), 
                                                 int(lim_xy[i] + inset_pad/2)), 
                                                 axis=axes[i])

    # plot chosen void in inset
    membership_slice[membership_slice > 0] = -1
    ax_zoom.imshow(membership_slice.mean(axis=axis).T, origin='lower',
                   cmap=cmap, vmin=-1, vmax=0.1,
                   extent=(inset_xy[0], inset_xy[0] + 2*inset_pad, 
                           inset_xy[1], inset_xy[1] + 2*inset_pad))

    mark_inset(ax, ax_zoom, loc1=2, loc2=1, ec=inset_color, alpha=0.8)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    if save_fn is not None:
        plt.savefig("Figure3.pdf", bbox_inches="tight")
    plt.show()

# optionally load from raw positions
# mesh_slice, voids_slice, membership_slice, inset_indx = load_slice(save_output=True)
# print('Inset void index:', inset_indx)

# load compressed plot data 
mesh_slice = np.load("mesh_slice.npy")
voids_slice = np.load("voids_slice.npy")
membership_slice = np.load("membership_slice.npy")

# make figure
plot_slice(mesh_slice, voids_slice, membership_slice, save_fn="Figure3.pdf")
