import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from napari import view_image
import warnings
from matplotlib import gridspec

try:
    from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD
    from sparkling.utils.plotting import scatter_shots
except ImportError:
    warnings.warn('SPARKLING not installed, not all features exist')

from .trajectory import view_gradients, view_sliced_traj, tf_sliced_dens, np_sliced_mask

def zoom_in(ax, inset_axes, trajectory, zoom_loc):
    pattern = trajectory.reshape(-1, trajectory.shape[-1])
    inset_axes.plot(pattern[:, 0], pattern[:, 1], c='b')
    inset_axes.plot(trajectory[0, :, 0], trajectory[0, :, 1], c='r')
    [x1, x2, y1, y2] = zoom_loc
    inset_axes.set_xlim(x1, x2)
    inset_axes.set_ylim(y1, y2)
    inset_axes.set_xticklabels('')
    inset_axes.set_yticklabels('')
    inset_axes.set_aspect('equal')
    new_ax = ax.indicate_inset_zoom(inset_axes, edgecolor="black", lw=2, alpha=1)
    
    
def show_trajectories(fig, shots, alpha=0.7, marker_sz=1, plot_samples=True, isnormalized=False):
    gs_traj = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.29)
    gss = gs_traj[0].subgridspec(1, 2, width_ratios=[2, 1], wspace=0, hspace=0)
    ax = fig.add_subplot(gss[0])
    plt_kwargs = {}
    if not isnormalized:
        shots = shots * 2 * np.pi
    num_shots = shots.shape[1]
    for i, shot in enumerate(shots):
        if i == 0:
            plt_kwargs['color'] = 'r'
            plt_kwargs['alpha'] = 1
        else:
            plt_kwargs['color'] = 'b'
            plt_kwargs['alpha'] = alpha
        if plot_samples:
            pltscat = ax.plot
            plt_kwargs['linewidth'] = marker_sz * (int(i == (num_shots - 1)) + 1)
        else:
            pltscat = ax.scatter
            plt_kwargs['s'] = marker_sz * (int(i == (num_shots - 1)) + 1)     
        pltscat(*shot.T, **plt_kwargs)
    limit = 1
    limits = [-limit, limit]
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"$k_x \rightarrow$", fontsize=12)
    ax.set_ylabel(r"$k_y \rightarrow$", fontsize=12)
    gss2 = gss[1].subgridspec(2, 1, hspace=0.2)
    if shots.shape[-1] == 3:
        ax.zaxis.set_ticklabels([])
        ax.set_zlim(limits)
        ax.set_zticks([-1, -0.5, 0, 0.5, 1])
        ax.set_zlabel(r"$k_z \rightarrow$", fontsize=12)
    for j, zoom_loc in enumerate([[0.06, 0.2, 0.06, 0.2], [-0.05, 0.05, -0.05, 0.05]]):
        axes_zoom = fig.add_subplot(gss2[j])
        zoom_in(ax, axes_zoom, shots, zoom_loc)