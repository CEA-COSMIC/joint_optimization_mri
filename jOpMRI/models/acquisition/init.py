from sparkling.constraints.utils import first_derivative_transpose, \
    first_derivative, proximity_L2, second_derivative

import numpy as np
global_iter = 0

INITIALIZATION = dict()
INITIALIZATION['scan_consts'] = {
    'gradient_raster_time': 0.010, 'gradient_mag_max': 40e-3,
    'slew_rate_max': 180e-3, 'gyromagnetic_constant': 42.576e3
}
INITIALIZATION['traj_params'] = {
    'dimension': 2, 'num_shots': 32, 'initialization': 'RadialCO',
    'num_samples_per_shot': 513, 'oversampling_factor': 5,
    'perturbation_factor': 0.75, 'num_revolutions': 2,
}
INITIALIZATION['recon_params'] = {
    'img_size': "320x320", 'FOV': "0.23x0.23"
}
INITIALIZATION['dist_params'] = {
    'cutoff': 25, 'decay': 2,
}
INITIALIZATION['algo_params'] = {
    'max_grad_iter': 5, 'max_proj_iter': 100, 'start_decim': 32,
    'proj_n_cpu': 1, 'stepDef': 1 / (2 * np.pi) * 1 / 16,
    'proj_gpu': True, 'shaking': False, 'tolerance': 0, 'proj_every_n_iter': 1,
    'remove_center_points': True,
    'fmm_params': {'fmm_method': 'gpu_direct_pykeops'}
}