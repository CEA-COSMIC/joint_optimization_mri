import numpy as np

import matplotlib.pyplot as plt
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD
from jOpMRI.models.acquisition.utils import _next_smooth_int
import tensorflow as tf
import napari


def view_gradients(trajectory, grads, log=False, return_figure=False, fig_scale=2, random_seed=None, num_shots=None, **plt_kwargs):
    fig = plt.figure(figsize=(6.4*fig_scale, 5*fig_scale))
    gradients = convert_NCxNSxD_to_NCNSxD(grads)
    shots = convert_NCxNSxD_to_NCNSxD(trajectory)
    if num_shots is not None and num_shots < shots.shape[0]:
        if random_seed is not None:
           np.random.seed(random_seed)
        idx = np.random.choice(shots.shape[0], num_shots, replace=False)
        shots = shots[idx]
        gradients = gradients[idx]
    if log:
        gradients[np.nonzero(np.abs(gradients) <= 1)] = 1
        gradients = np.log(np.abs(gradients)) * np.sign(gradients)
    if shots.shape[-1] == 2:
        plt.quiver(shots.T[0], shots.T[1], -gradients.T[0], - gradients.T[1], angles='xy', **plt_kwargs)
    else:
        ax = fig.add_subplot(projection='3d')
        ax.quiver(
            shots.T[0],
            shots.T[1],
            shots.T[2],
            -gradients.T[0],
            -gradients.T[1],
            -gradients.T[2],
            **plt_kwargs,
        )
    return fig


def view_at_traj_values(trajectory, values, **plt_kwargs):
    shots = convert_NCxNSxD_to_NCNSxD(trajectory)
    plt.scatter(shots[:, 0], shots[:, 1], c=values, cmap='gray')


def tf_sliced_dens(shots, vol_shape=(256, 256, 224), osf=1):
    import tensorflow_nufft as tfft
    grid_shape = tf.TensorShape([_next_smooth_int(i*osf) for i in vol_shape])
    interp = tfft.spread(tf.ones(shots.shape[0], dtype=tf.complex64), shots*2*np.pi, grid_shape, tol=1e-3)
    return interp


def np_sliced_mask(samples_locations, img_shape=(256, 256, 224)):
    if samples_locations.shape[-1] != len(img_shape):
        raise ValueError("Samples locations dimension doesn't correspond to ",
                         "the dimension of the image shape")
    locations = np.copy(samples_locations).astype("float")
    locations += 0.5
    if np.any(locations >= 1):
        locations = np.delete(locations, np.nonzero(np.any(locations >= 1 , axis=1)), axis=0)
    locations *= img_shape
    locations = np.floor(locations).astype('int')
    loc = np.unique(locations, axis=0)
    mask = np.zeros(img_shape, dtype="int")
    mask[tuple(loc.T)] = 1
    return mask

def view_sliced_traj(model, osf=1, log=False, use_tf=False):
    shots = model.base_traj.trajectory.numpy()
    shots = shots.reshape(-1, shots.shape[-1])
    vol_shape = model.image_size
    if use_tf:
        interp = tf_sliced_dens(shots, vol_shape, osf)
        log = True
    else:
        vol_shape = tuple([shape*osf for shape in vol_shape])
        interp = np_sliced_mask(shots*np.pi, vol_shape)
    interp = np.abs(interp)
    if log:
        interp = np.log(interp)
    napari.view_image(interp)
