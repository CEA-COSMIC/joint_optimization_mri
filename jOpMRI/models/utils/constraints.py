import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class ClipValue(Constraint):
    def __init__(self, min_value=-1/2/np.pi, max_value=1/2/np.pi):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)


class ScannerConstraints(Constraint):
    def __init__(self, scan_consts, recon_params, oversampling_factor,
                 algorithm='primal_dual', min_value=-1/2/np.pi,
                 max_value=1/2/np.pi, num_iterations=100):
        # Get sparkling imports
        from sparkling.utils.constraints import get_kinetic_constraints, decimate_kinetic_constraints
        from sparkling.constraints.constraints import Linear
        from sparkling.optimization.projection import Project_Curve_Affine_Constraints
        from sparkling.parameters.initializations.common import INITIALIZATION
        self.uses_cupy = False
        try:
            import cupy as cp
            self.xp = cp
            self.uses_cupy = True
        except:
            self.xp = tf.experimental.numpy
        self.get_kinetic_constraints = get_kinetic_constraints
        self.decimate_kinetic_constraints = decimate_kinetic_constraints
        self.Project_Curve_Affine_Constraints = Project_Curve_Affine_Constraints
        self.Linear = Linear

        self.max_value = max_value
        self.min_value = min_value
        self.base_kinetic_constraints = get_kinetic_constraints(
            scan_consts=scan_consts,
            recon_params=recon_params,
            kinetic_constraint_init=INITIALIZATION['kinetic_constraint_init'],
            oversampling_factor=oversampling_factor,
        )
        self.algorithm = algorithm
        self.num_iterations = num_iterations
        self.inv_lipschitz_constant = None
        self.kinetic_constraints = None
        self.linear_constraints = None

    def lipschitz_constant_op(self, data):
        result = self.kinetic_constraints[0].linear_adj_op(
            self.kinetic_constraints[0].linear_op(data)
        )
        for constraint in self.kinetic_constraints[1:]:
            result += constraint.linear_adj_op(constraint.linear_op(data))
        return result

    def update_constraints(self, decim, Ns, mask, fixed_points, D=2):
        self.kinetic_constraints = self.decimate_kinetic_constraints(self.base_kinetic_constraints, decim)
        self.linear_constraints = self.Linear(
            Ns,
            D,
            {'fixed_points': [mask, fixed_points]},
            True
        )

    def setup_lips(self, input_shape):
        from modopt.math.matrix import PowerMethod
        Nc, Ns, D = input_shape
        pm = PowerMethod(
            self.lipschitz_constant_op,
            (1, Ns, D),
            data_type=np.float32,
        )
        self.inv_lipschitz_constant = pm.inv_spec_rad * 0.8

    def _project(self, shots):
        if self.inv_lipschitz_constant is None:
            self.setup_lips(shots.shape)
        if self.uses_cupy:
            shots = self.xp.fromDlpack(tf.experimental.dlpack.to_dlpack(shots))
        else:
            shots = self.xp.array(shots)
        projected_shots = self.Project_Curve_Affine_Constraints(
            shots,
            self.num_iterations,
            self.inv_lipschitz_constant,
            self.kinetic_constraints,
            linear_constraints=self.linear_constraints,
            compute_backend='cupy',
            progress=False,
        )
        if self.uses_cupy:
            projected_shots = tf.experimental.dlpack.from_dlpack(projected_shots.toDlpack())
        return projected_shots

    def get_grads_n_slew(self, shots):
        grads = self.kinetic_constraints[0].linear_op(shots)
        slew = self.kinetic_constraints[1].linear_op(shots)
        return grads, slew

    def __call__(self, shots):
        if self.kinetic_constraints == None:
            raise ValueError('Please update the constraints at least once!')
        projected = tf.py_function(self._project, [shots], Tout=tf.float32)
        projected = tf.clip_by_value(projected, self.min_value, self.max_value)
        projected.set_shape(shots.shape)
        return projected
