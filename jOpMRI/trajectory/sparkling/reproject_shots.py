
# TODO Move this to sparkling!
def reproject(file, INIT, num_projection=1000):
    import numpy as np
    from modopt.math.matrix import PowerMethod

    from sparkling.utils.gradient import get_kspace_loc_from_gradfile
    from sparkling.optimization.projection import project_fast_fista
    from sparkling.constraints.constraints import Kinetic, Linear
    from sparkling.constraints.utils import first_derivative_transpose, \
        first_derivative, proximity_L2, second_derivative

    # Setup projector
    speed = {'function': proximity_L2, 'operator': first_derivative,
             'operatorT': first_derivative_transpose}
    acceleration = {'function': proximity_L2, 'operator': second_derivative,
                    'operatorT': second_derivative}
    kinetic_constraints = [
        Kinetic(**speed, bound=np.array(
            INIT['traj_params']['dimension'] * (
                INIT['scan_consts']['gradient_mag_max'] *
                INIT['scan_consts']['gyromagnetic_constant'] *
                INIT['scan_consts']['gradient_raster_time'],
            )
        )), Kinetic(**acceleration, bound=np.array(
            INIT['traj_params']['dimension'] * (
                INIT['scan_consts']['slew_rate_max'] *
                INIT['scan_consts']['gyromagnetic_constant'] *
                INIT['scan_consts']['gradient_raster_time'] *
                INIT['scan_consts']['gradient_raster_time'],
            )
        ))
    ]

    def lipschitz_constant_op(data):
        result = kinetic_constraints[0].opT(
            kinetic_constraints[0].op(data))
        for k in np.arange(1, len(kinetic_constraints)):
            result += kinetic_constraints[k].opT(
                kinetic_constraints[k].op(data))
        return result

    def compute_inv_lipschitz_constant(shots_shape):
        num_samples = shots_shape[1]
        dimension = shots_shape[2]
        pm = PowerMethod(lipschitz_constant_op,
                         (1, num_samples, dimension))
        return pm.inv_spec_rad

    shots, params = get_kspace_loc_from_gradfile(
        file,
        INIT['scan_consts']['gradient_raster_time'],
        INIT['traj_params']['num_samples_per_shot'],
    )
    inv_lipschitz_constant = compute_inv_lipschitz_constant(shots.shape)
    linear_constraints = Linear(
        shots.shape[1],
        shots.shape[2],
        {"TE_point": int(
            np.ceil(INIT['traj_params']['num_samples_per_shot']/2))},
        True
    )
    shots_new = project_fast_fista(
        shots=shots,
        num_iterations=num_projection,
        inv_lipschitz_constant=inv_lipschitz_constant * 0.8,
        kinetic_constraints=kinetic_constraints,
        linear_constraints=linear_constraints,
        use_gpu=True,
        return_gpu=False,
        verbose=1,
    )
    gradients = np.diff(shots_new, axis=1) / (
        INIT['scan_consts']['gyromagnetic_constant'] *
        INIT['scan_consts']['gradient_raster_time']
    )
    slew = np.diff(gradients, axis=1) / \
        INIT['scan_consts']['gradient_raster_time']
    first_kspace_point = shots_new[:, 0, :]
    return gradients, first_kspace_point, slew
