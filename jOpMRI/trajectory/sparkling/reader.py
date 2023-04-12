import tensorflow as tf
import numpy as np

from jOpMRI.models.acquisition.utils import get_density_compensators

from sparkling.utils.gradient import get_kspace_loc_from_gradfile
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD


def read_trajectory(filename, osf=5, dcomp=False, gradient_raster_time=0.01, interpob=None, return_reshaped=True):
    trajectory, params = get_kspace_loc_from_gradfile(
        filename,
        gradient_raster_time=gradient_raster_time,
        dwell_time=gradient_raster_time/osf,
        read_shots=True,
    )
    if return_reshaped:
        trajectory = convert_NCxNSxD_to_NCNSxD(trajectory)
        trajectory = tf.convert_to_tensor(trajectory.T, dtype=tf.float32)
        trajectory = trajectory[None, ...]
    trajectory = trajectory * 2 * params['FOV'] / params['img_size'] * np.pi
    if dcomp:
        if interpob is None:
            raise ValueError('Please send interpob for getting density_comp')
        density_comp = get_density_compensators(trajectory[0], interpob)
        return trajectory, density_comp
    return trajectory
