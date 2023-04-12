import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.interpolate


def interpolate_shots_only(trajectory, target_Ns, method='linear'):
    traj = tfp.math.interp_regular_1d_grid(
        x=tf.cast(tf.linspace(0, 1, target_Ns), trajectory.dtype),
        x_ref_min=0,
        x_ref_max=1,
        y_ref=trajectory,
        axis=1
    )
    traj = tf.clip_by_value(traj, -1/2/np.pi, 1/2/np.pi)
    return traj


def sp_interpolate_shots(trajectory, target_Ns, method='linear'):
    traj = scipy.interpolate.interp1d(
            x=tf.linspace(0, 1, trajectory.shape[1]),
            y=trajectory,
            kind=method,
            axis=1,
            bounds_error=False,
            fill_value=0
        )(tf.linspace(0, 1, target_Ns)).astype(np.float32)
    traj = tf.clip_by_value(traj, -1/2/np.pi, 1/2/np.pi)
    return traj


def get_grads_n_slew(shots, gyromagnetic_constant, gradient_raster_time, FOV, img_size):
    shots = shots * (np.asarray(img_size) / 2 / np.asarray(FOV)).astype(np.float32)
    gradients = tf.experimental.numpy.diff(shots, axis=1) / (gyromagnetic_constant * gradient_raster_time)
    slew_rate = tf.experimental.numpy.diff(gradients, axis=1) / gradient_raster_time
    gradients = tf.norm(gradients, axis=-1)
    slew_rate = tf.norm(slew_rate, axis=-1)
    return gradients, slew_rate, shots[:, 0]
