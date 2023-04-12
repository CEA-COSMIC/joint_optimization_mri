import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import warnings

from jOpMRI.trajectory.utils import interpolate_shots_only, get_grads_n_slew
from jOpMRI.models.utils.constraints import ClipValue, ScannerConstraints
from .utils import get_density_compensators, SampleRepulsion, ConstraintPenalty

from modopt.base.backend import move_to_cpu
from sparkling.utils.trajectory import init_shots, interpolate_shots
import logging

log = logging.getLogger(__name__)


class _Trajectory(Layer):
    def __init__(self, traj_params, scan_consts, recon_params, algo_params, dcomp=True, interpob=None,
                 img_size=(320, 320), dist_params={}, traj_init=None, prospective=False,
                 nufft_implementation='tensorflow_nufft', batch_size=1, **kwargs):
        super(_Trajectory, self).__init__(**kwargs)
        self.traj_params = traj_params
        self.algo_params = algo_params
        self.recon_params = recon_params
        self.scan_consts = scan_consts
        self.batch_size = batch_size
        self.interpob = interpob
        self.dcomp = dcomp
        self.project = algo_params['max_proj_iter'] > 0
        self.oversampling_factor = traj_params['oversampling_factor']
        self.prospective = prospective
        self.nufft_implementation = nufft_implementation
        if 'trajectory_regularization' in self.algo_params.keys():
            if self.algo_params['trajectory_regularization'] == 'temporal_repulsion':
                self.regularizer = SampleRepulsion(temporal_repulsion=True)
            elif self.algo_params['trajectory_regularization'] == 'repulsion':
                self.regularizer = SampleRepulsion(temporal_repulsion=False)
            elif self.algo_params['trajectory_regularization'] != 'constraint_penalty':
                self.regularizer = None
        else:
            self.regularizer = None
        if 'IO' in traj_params['initialization']:
            self.TE_point = 0.5
        else:
            self.TE_point = 0
        if traj_params['initialization'] != 'Random':
            self.init, self.mask_init = init_shots(**traj_params)
        else:
            self.init = np.random.uniform(
                -1/2/np.pi,
                1/2/np.pi,
                (traj_params['num_shots'], traj_params['num_samples_per_shot'], traj_params['dimension']),
            )
            self.mask_init = np.zeros(self.init.shape[:-1], dtype=np.bool)
        self.mask = self.mask_init
        if self.project:
            self.projector = ScannerConstraints(
                scan_consts=scan_consts,
                recon_params=recon_params,
                oversampling_factor=self.oversampling_factor,
                num_iterations=algo_params['max_proj_iter'],
            )
            self.projector.update_constraints(
                decim=self.current_decim,
                Ns=self.traj_params['num_samples_per_shot'],
                mask=self.mask_init,
                fixed_points=self.init[self.mask_init],
            )
        else:
            self.projector = ClipValue()
        self.img_size = img_size
        if self.project:
            # FIXME this is very hacky!
            self.projector.uses_cupy = False
            init_shot = move_to_cpu(self.projector._project(self.init))
            self.projector.uses_cupy = True
        else:
            init_shot = self.init
        self._init_trajectory(tf.clip_by_value(init_shot, -1/2/np.pi, 1/2/np.pi))
        self.current_traj = None
        self.send_traj_n_dcomp = tf.function(self._send_traj_n_dcomp)

    def build(self, input_shape):
        super(_Trajectory, self).build(input_shape)

    def _init_trajectory(self, init_shot):
        self.trajectory = self.add_weight(
            shape=init_shot.shape,
            name='BaseTrajectory',
            dtype=tf.float32,
            constraint=self.projector,
            regularizer=self.regularizer,
        )
        self.trajectory.assign(tf.cast(init_shot, tf.float32))

    @tf.custom_gradient
    def _shape_grad_log(self, x):
        def grad(dy):
            # We multiply by batch size for scaling right
            grad_1 = tf.maximum(tf.abs(dy), tf.cast(1, tf.float32))
            return tf.math.log(grad_1) * tf.math.sign(dy)
        return x, grad

    def _send_traj_n_dcomp(self, target_Ns=None, batch_size=1):
        # By pass the trajectory and density compensation estimation stage
        # if we arent learning the trajectory
        if self.trainable == True or self.current_traj == None:
            if 'shape_grad' in self.algo_params and self.algo_params['shape_grad'] is not None:
                if self.algo_params['shape_grad'] == 'log':
                    traj = self._shape_grad_log(self.trajectory)
                else:
                    raise ValueError('Cant understand how to shape trajectory gradients')
            else:
                traj = self.trajectory
            Nc, Ns, D = traj.shape
            if target_Ns == None:
                target_Ns = Ns
            if Ns != target_Ns or self.oversampling_factor !=1:
                # Interpolate the trajectory
                traj = interpolate_shots_only(traj, target_Ns * self.oversampling_factor)
            if self.prospective:
                traj = traj[:, :-self.oversampling_factor, :]
                target_Ns = target_Ns - 1
            self.current_traj = tf.transpose(tf.reshape(
                traj,
                (Nc * target_Ns * self.oversampling_factor, D)
            )) * 2 * np.pi * np.pi
            if self.dcomp:
                self.current_DC = get_density_compensators(
                    self.current_traj,
                    self.img_size,
                    interpob=self.interpob,
                    implementation=self.nufft_implementation
                )
            if self.nufft_implementation == 'tfkbnufft':
                self.current_traj = tf.repeat(self.current_traj[None, :], self.batch_size, axis=0)
        if self.dcomp:
            return self.current_traj, self.current_DC
        return self.current_traj

    def get_grads_n_slew(self):
        # Always call this AFTER calling the trajectory!
        traj = tf.reshape(
            tf.transpose(self.current_traj),
            (tf.shape(self.trajectory)[0], -1, tf.shape(self.trajectory)[2])
        ) / np.pi 
        return get_grads_n_slew(
            traj,
            img_size=self.recon_params['img_size'],
            FOV=self.recon_params['FOV'],
            gyromagnetic_constant=self.scan_consts['gyromagnetic_constant'],
            gradient_raster_time=self.scan_consts['gradient_raster_time'],
        )
    
    def call(self, input):
        return self.send_traj_n_dcomp(input)


class MultiResolution(_Trajectory):
    """
    Layer to handle multi-resolution of trajectory
    Attributes
    ----------
    start_decim: int,
        Holds the starting decimation value.
    current_decim: int,
        The current decimation level.
    """

    def __init__(self, traj_params, algo_params, interpolate='linear', **kwargs):
        """
        Parameters
        ----------
        start_decim: int, default 64
            The starting decimation level for multi-resoultion framework for trajectory.
        interpolate: str, default 'linear'
            The kind of interpolation to scale up the number of samples in trajectory.
            `linear`: Linearly interpolate data
            `none`: No Interpolation, the downsampled trajectory is passed as output.
        **kwargs: dict
            Extra arguments to base class and layer class
        """
        self.start_decim = algo_params['start_decim']
        self.current_decim = self.start_decim
        self.interpolate = interpolate
        self.target_Ns = traj_params['num_samples_per_shot']
        traj_params['num_samples_per_shot'] = int(self.target_Ns / self.current_decim)
        self.penalty_loss = ConstraintPenalty(
                img_size=kwargs['recon_params']['img_size'],
                FOV=kwargs['recon_params']['FOV'],
                gyromagnetic_constant=kwargs['scan_consts']['gyromagnetic_constant'],
                gradient_raster_time=kwargs['scan_consts']['gradient_raster_time'],
                GMax=kwargs['scan_consts']['gradient_mag_max'],
                SMax=kwargs['scan_consts']['slew_rate_max'],
                scale=algo_params['gradient_penalty_scale'] if 'gradient_penalty_scale' in algo_params else 0.01,
                num_samples_per_shot=self.target_Ns,
            )
        if not traj_params['num_samples_per_shot'] % 2:
            traj_params['num_samples_per_shot'] = traj_params['num_samples_per_shot'] + 1
        if 'trajectory_regularization' in algo_params and algo_params['trajectory_regularization'] == 'constraint_penalty':
            self.regularizer = self.penalty_loss
        super(MultiResolution, self).__init__(
            traj_params=traj_params,
            algo_params=algo_params,
            **kwargs,
        )
        print('Optimizing for Ns : ' + str(self.target_Ns))

    def upscale(self, factor=2):
        """
        Upscale the problem by a factor
        Parameters
        ----------
        factor: int, default 2
            The factor by which, the trajectory must be upscaled.
        """
        self.current_decim = self.current_decim // factor
        new_shots, self.mask = interpolate_shots(
            self.trajectory,
            self.target_Ns // self.current_decim,
            mask=self.mask, 
            img_size=self.img_size, 
            OSF=1,
        )
        Ns = new_shots.shape[1]
        if self.project:
            self.projector.update_constraints(
                decim=self.current_decim,
                Ns=Ns,
                mask=self.mask,
                fixed_points=new_shots[self.mask],
            )
        log.info("Upscaling by factor of " + str(factor) + " to Ns = " + str(Ns) + " Decimation step: " + str(self.current_decim))
        self._init_trajectory(new_shots)
        # Retrace as the variable changed!
        self.send_traj_n_dcomp = tf.function(self._send_traj_n_dcomp)

    def downscale(self, factor=2):
        self.current_decim = self.current_decim * factor
        new_shots = self.trajectory[:, ::factor, :]
        self.mask = self.mask[:, ::factor]
        Ns = new_shots.shape[1]
        if self.project:
            self.projector.update_constraints(
                decim=self.current_decim,
                Ns=Ns,
                mask=self.mask,
                fixed_points=new_shots[self.mask],
            )
        self._init_trajectory(new_shots)
        # Retrace as the variable changed!
        self.send_traj_n_dcomp = tf.function(self._send_traj_n_dcomp)

    def get_penalty_losses(self):
        return self.penalty_loss.get_all_loss(self.trajectory)
    
    def call(self, input):
        # target_Ns ensures that overall trajectory is always Ns
        return self.send_traj_n_dcomp(self.target_Ns, input)

