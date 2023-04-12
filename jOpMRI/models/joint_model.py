from genericpath import exists
import os
import pickle as pkl
import shutil
import tensorflow as tf
import numpy as np
import logging
import h5py
import git

from jOpMRI.models.preprocess.coil_combination import VirtualCoilCombination
from jOpMRI.models.acquisition.trajectory import MultiResolution
from jOpMRI.models.acquisition.forward import BasicForward, ConstantB0T2
from jOpMRI.models.reconstruction.adjoint import Adjoint
from jOpMRI.models.reconstruction.stack_vnet import StackVNet
from jOpMRI.models.utils.compile import default_multiopt_compile, load_optimizer
from jOpMRI.models.utils.schdulers import get_total_epochs
from jOpMRI.models.utils.gradient_accumulation import GAModelWrapper
from jOpMRI.trajectory.sparkling.reader import read_trajectory
from jOpMRI.data.utils import phase_correction

from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
from fastmri_recon.models.subclassed_models.unet import UnetComplex

from tf_fastmri_data.preprocessing_utils.extract_smaps import non_cartesian_extract_smaps

from jOpMRI.models.acquisition import INITIALIZATION as INIT

from tfkbnufft.kbnufft import KbNufftModule, kbnufft_adjoint


log = logging.getLogger(__name__)


class AcqRecModel(GAModelWrapper):
    def __init__(self, trajectory_kwargs=INIT, acq_kwargs={}, recon_kwargs={}, combine_coil=False, batch_size=None,
                 multicoil=False, prospective=False, opt_kwargs={}, nufft_implementation='tensorflow-nufft', cfg=None, 
                 accum_steps=1, **kwargs):
        super(AcqRecModel, self).__init__(accum_steps=accum_steps, **kwargs)
        self.cfg = cfg
        self.trajectory_kwargs = trajectory_kwargs.copy()
        self.image_size = tuple(trajectory_kwargs['recon_params']['img_size'])
        self.dcomp = recon_kwargs['dcomp']
        self.combine_coil = combine_coil
        self.opt_kwargs = opt_kwargs
        self.recon_kwargs = recon_kwargs
        self.multicoil = multicoil
        self.prospective = prospective
        self.nufft_implementation = nufft_implementation
        self.continue_run = False
        if self.nufft_implementation == 'tfkbnufft':
            if batch_size is None:
                raise ValueError("Need batch_size for tfkbnufft")
            self.nufft_op = KbNufftModule(
                im_size=self.image_size,
                grid_size=None,
                norm='ortho',
                grad_traj=True,
            )
            self.interpob = self.nufft_op._extract_nufft_interpob()
        else:
            self.interpob = None
        self.batch_size = batch_size
        if acq_kwargs['type'] == 'basic':
            self.foward_model = BasicForward(
                interpob=self.interpob,
                nufft_implementation=nufft_implementation,
                name='forward_model',
            )
        elif acq_kwargs['type'] == 'constant_t2b0':
            self.foward_model = ConstantB0T2(
                interpob=self.interpob,
                Nc=trajectory_kwargs['traj_params']['num_shots'],
                Ns=trajectory_kwargs['traj_params']['num_samples_per_shot'],
                gradient_raster_time=trajectory_kwargs['scan_consts']['gradient_raster_time'],
                osf=trajectory_kwargs['traj_params']['oversampling_factor'],
                TE=trajectory_kwargs['traj_params']['num_samples_per_shot']//2 if 'IO' in trajectory_kwargs['traj_params']['initialization'] else 1,
                nufft_implementation=nufft_implementation,
                **acq_kwargs['params'],
                name='forward_model'
            )
        else:
            raise ValueError('Value of forward model is wrong!')
        self.base_traj = MultiResolution(
            dcomp=self.dcomp,
            img_size=self.image_size,
            name='base_trajectory',
            prospective=self.prospective,
            batch_size=self.batch_size,
            interpob=self.interpob,
            nufft_implementation=nufft_implementation,
            **trajectory_kwargs,
        )
        if self.recon_kwargs['type'] == 'adjoint':
            self.recon_net = Adjoint(
                self.interpob,
                img_size=self.image_size,
                dcomp=recon_kwargs['dcomp'],
                multicoil=multicoil,
                nufft_implementation=nufft_implementation,
                name='reconstruction_model',
            )
        elif self.recon_kwargs['type'] == 'ncpdnet':
            self.recon_net = NCPDNet(
                multicoil=multicoil,
                im_size=self.image_size,
                dcomp=recon_kwargs['dcomp'],
                name='reconstruction_model',
                grad_traj=True,
                nufft_implementation=nufft_implementation,
                **recon_kwargs['params'],
            )
        elif self.recon_kwargs['type'] == 'vnet_stacked':
            self.recon_net = StackVNet(
                **recon_kwargs['params'],
                vnet_kwargs={
                    'im_size': self.image_size,
                    'dcomp': recon_kwargs['dcomp'],
                    'grad_traj': True,
                    'nufft_implementation': nufft_implementation,
                    **recon_kwargs['vnet_kwargs'],
                }
            )
        elif self.recon_kwargs['type'] == 'unet':
            self.recon_net = UnetComplex(
                im_size=self.image_size,
                dcomp=recon_kwargs['dcomp'],
                multicoil=multicoil,
                dealiasing_nc_fastmri=True,
                grad_traj=True,
                nufft_implementation=nufft_implementation,
                name='reconstruction_model',
                **recon_kwargs['params'],
            )
        else:
            raise ValueError('Value of recon_type is wrong')
        self.dc_adjoint = Adjoint(
            self.interpob,
            img_size=self.image_size,
            dcomp=recon_kwargs['dcomp'],
            nufft_implementation=nufft_implementation,
            name='dc_adjoint',
            multicoil=self.multicoil
        )
        self.use_recon_adjoint = False
        if self.combine_coil:
            self.coil_combiner = VirtualCoilCombination(name='virtual_coil_combiner')
        self.forward_type = acq_kwargs['type']
        self.recon_type = recon_kwargs['type']
        if recon_kwargs['type'] != 'adjoint' and recon_kwargs['weights_file'] is not None:
            self.init_recon(recon_kwargs['weights_file'])
        self.traj_loaded = False
        num_steps_per_epoch = cfg['train']['num_steps_per_epoch']
        if not isinstance(num_steps_per_epoch, int):
            num_steps_per_epoch = num_steps_per_epoch[0]
        # We need a different optimizer for penalty steps as it needs to be varied with upsampling
        self.only_traj_related_learn = 'only_traj_related_learn' in cfg['train'] and cfg['train']['only_traj_related_learn']
        default_multiopt_compile(
            self,
            total_decim=cfg['trajectory']['algo_params']['start_decim'] * (
                cfg['train']['decay_extra_factor_over_decim'] if 'decay_extra_factor_over_decim' in cfg['train'] else 1
            ),
            total_steps=get_total_epochs(cfg) * num_steps_per_epoch,
            **self.opt_kwargs
        )
        self.set_initializers()
        
    def init_recon(self, weights=None):
        if weights == None:
            raise ValueError("The model file was not passed")
        # Call the model once
        self.call(tf.zeros((1, 1, *self.image_size)))
        if self.recon_kwargs['type'] == 'ncpdnet' or 'unet':
            model_weights = h5py.File(weights, 'r')
            self.upscale_to_match_Ns(model_weights['base_trajectory']['BaseTrajectory:0'].shape[1])
            y = self(tf.cast(tf.zeros((1, 1, *self.image_size)), tf.complex64))
            self.load_weights(weights)
            # Downscale back the problem
            self.base_traj.downscale(self.trajectory_kwargs['algo_params']['start_decim'])
        else:
            self.recon_net.load_weights(weights)
        self.recon_weights = weights
        # By default, we dont learn in case we load a network.
        self.recon_net.trainable = self.recon_kwargs['trainable']

    def upscale_to_match_model_file(self, model_file, remove_last_point=False):
        model_weights = h5py.File(model_file, 'r')
        self.upscale_to_match_Ns(model_weights['base_trajectory']['BaseTrajectory:0'].shape[1])

    def temp_load_weights_no_GA(self, model_file):
        temp_copy = os.path.join(os.path.dirname(model_file), 'temp.h5')
        shutil.copyfile(model_file, temp_copy)
        f = h5py.File(temp_copy, 'a')
        # Remove GA weights
        del f['top_level_model_weights']
        f.close()
        self.load_weights(temp_copy)
        log.info("Successfully loaded weights from {} without GA".format(model_file))
    
    def continue_load_weights(self, model_file, load_opt=True):
        self.upscale_to_match_model_file(model_file)
        try:
            self.load_weights(model_file)
        except:
            log.warn("Could not load weights from file. Trying to load weights ignoring GA")
            self.temp_load_weights_no_GA(model_file)
        default_multiopt_compile(self, **self.opt_kwargs)
        if load_opt:
            folder = os.path.dirname(model_file)
            epoch = int(os.path.basename(model_file).split('_')[1][1:])
            file = 'optimizer_E{epoch:02d}.pkl'.format(epoch=epoch)
            if not exists(os.path.join(folder, file)):
                # We used to only save last optimizer before!
                file = 'optimizer.pkl'
            load_optimizer(self, self.opt_kwargs, os.path.join(folder, file))
        self.continue_run = True

    def init_trajectory(self, traj_file, downsample_traj=False):
        trajectory = read_trajectory(
            traj_file,
            osf=1,
            dcomp=False,
            gradient_raster_time=self.trajectory_kwargs['scan_consts']['gradient_raster_time'],
            return_reshaped=False,
        )
        trajectory = trajectory / np.pi / np.pi / 2
        if downsample_traj:
            self.base_traj.set_weights([trajectory[:, ::self.trajectory_kwargs['algo_params']['start_decim'], :]])
            # Make current traj None
            self.base_traj.current_traj = None
        else:
            self.upscale_to_match_Ns(trajectory.shape[1])
            self.base_traj.set_weights([trajectory])
        self.traj_loaded = True

    def upscale_to_match_Ns(self, Ns):
        start_decim = self.base_traj.current_decim
        if Ns != self.base_traj.trajectory.shape[1]:
            possible_decim = np.array([2**i for i in range(int(np.log2(start_decim)) + 1)])
            decim = possible_decim[np.argmin(np.abs(
                possible_decim - start_decim * self.base_traj.trajectory.shape[1] / Ns))
            ]
            if decim != self.base_traj.current_decim:
                self.base_traj.upscale(self.base_traj.current_decim // decim)
        # Make current traj None
        self.base_traj.current_traj = None

    def get_traj_input(self, input):
        if self.dcomp:
            return self.base_traj(input)
        else:
            return (self.base_traj(input), )

    def preprocess_n_recon(self, kspace, trajectory, density_comp=None):
        more_inputs = tuple()
        extra_inputs = tuple()
        if self.dcomp:
            if self.multicoil:
                Smaps = non_cartesian_extract_smaps(
                    kspace,
                    trajectory,
                    density_comp[None, ...],
                    kbnufft_adjoint(self.interpob),
                    self.image_size,
                    low_freq_percentage=30,
                )
                more_inputs += (Smaps,)
            if not self.use_recon_adjoint and (self.recon_type in ['ncpdnet', 'unet', 'vnet_stacked']):
                if len(self.image_size) == 2:
                    extra_inputs += (([self.image_size[0]]), )
                else:
                    extra_inputs += (self.image_size, )                    
                density_comp = density_comp[None, ...]
                kspace = kspace[..., None]
            extra_inputs += (tf.cast(density_comp, kspace.dtype),)
        recon_inputs = [kspace, trajectory, *more_inputs, extra_inputs]
        if self.use_recon_adjoint:
            recon = self.dc_adjoint(recon_inputs)
        else:
            recon = self.recon_net(recon_inputs)
        if not self.use_recon_adjoint and (self.recon_type in ['ncpdnet', 'unet', 'vnet_stacked']):
            recon = recon[:, None, ...]
            if self.recon_type == 'ncpdnet':
                recon = recon[..., 0]
        if self.cfg['train']['optimizer']['loss'] == 'compound_l1_mssim_loss_with_penalty':
            return [recon, *self.base_traj.get_grads_n_slew()]
        return recon

    def call(self, inputs):
        if self.combine_coil:
            inputs = self.coil_combiner(inputs[0])[None, :]
        if self.prospective:
            input, kspace_shifts = inputs
        else:
            input = inputs
        if self.combine_coil:
            image = self.coil_combiner(image)[None, :]
        batch_size = tf.shape(input)[0]
        traj_input = self.get_traj_input(batch_size)
        if self.only_traj_related_learn:
            return traj_input[0]
        if not self.prospective:
            image = input
            kspace = self.foward_model([image, traj_input[0]])
        else:
            kspace = input
            kspace_shifts = kspace_shifts / 1000 / self.trajectory_kwargs['recon_params']['FOV'] * self.image_size
            kspace = phase_correction(traj_input[0], kspace, kspace_shifts)
        return self.preprocess_n_recon(kspace, *traj_input)

    def set_initializers(self, start_steps=0):
        self.start_steps = start_steps
        self.h_epochs = []
        self.actual_history = {}
        repo = git.Repo(os.path.join(__file__, "..", "..", ".."))
        sha = repo.head.object.hexsha
        try:
            with open(os.path.join(os.getcwd(), '.hydra', 'git_sha.txt'), 'w') as f:
                f.write(sha)
        except:
            pass

    def copy_history(self):
        self.history.epoch = self.h_epochs
        self.history.history = self.actual_history

    def trajectory_fit(self, reuse_traj_opt=False, reuse_recon_opt=False,  shards=1, index=0, **kwargs):
        if not self.traj_loaded and self.trajectory_kwargs['traj_init'] is not None:
            self.init_trajectory(self.trajectory_kwargs['traj_init'], downsample_traj=True)
        if self.use_recon_adjoint:
            log.info('Fitting only trajectory with DC Adjoint')
        else:
            log.info('Fitting only trajectory with {} reconstructor'.format(self.recon_type))
        self.recon_net.trainable = False
        self.base_traj.trainable = True
        default_multiopt_compile(self, reuse_traj_opt=reuse_traj_opt, reuse_recon_opt=reuse_recon_opt, **self.opt_kwargs)
        self.continue_fit(shards=shards, index=index, **kwargs)

    def recon_net_fit(self, reuse_traj_opt=False, reuse_recon_opt=False, shards=1, index=0, start_steps=None, **kwargs):
        if start_steps is not None:
            self.set_initializers(start_steps=start_steps)
        if not self.traj_loaded and self.trajectory_kwargs['traj_init'] is not None:
            self.init_trajectory(self.trajectory_kwargs['traj_init'])
        log.info('Fitting only {} reconstructor'.format(self.recon_type))
        self.recon_net.trainable = True
        self.base_traj.trainable = False
        default_multiopt_compile(self, reuse_traj_opt=reuse_traj_opt, reuse_recon_opt=reuse_recon_opt, **self.opt_kwargs)
        self.continue_fit(shards=shards, index=index, **kwargs)

    def joint_fit(self, reuse_traj_opt=False, reuse_recon_opt=False, shards=1, index=0, **kwargs):
        if not self.traj_loaded and self.trajectory_kwargs['traj_init'] is not None:
            self.init_trajectory(self.trajectory_kwargs['traj_init'], downsample_traj=True)
        log.info('Moving to jointly optimizing with {} reconstructor'.format(self.recon_type))
        self.recon_net.trainable = True
        self.base_traj.trainable = True
        default_multiopt_compile(self, reuse_traj_opt=reuse_traj_opt, reuse_recon_opt=reuse_recon_opt, **self.opt_kwargs)
        self.continue_fit(**kwargs, index=index, shards=shards)

    def continue_fit(self, x, epochs, index, shards, validation_data=None, left_over=None, **kwargs):
        log.info(
            'Continue to fit for {} epochs with {} steps/epoch at step {} with {}-index of {} shards of data'
                .format(epochs, kwargs['steps_per_epoch'], self.start_steps, index, shards)
        )
        if left_over is not None and self.continue_run:
            log.info("Continuing from previous run, from {} epochs, doing just {} epochs now".format(self.start_steps, left_over))
            epochs = left_over
        if validation_data is not None:
            kwargs['validation_data'] = validation_data.shard(shards, index)
        self.reinit_grad_accum()
        self.fit(
            x=x.shard(shards, index=index).prefetch(1),
            initial_epoch=self.start_steps,
            epochs=self.start_steps + epochs,
            **kwargs,
        )
        self.h_epochs += self.history.epoch
        for h, v in self.history.history.items():
            if h not in self.actual_history.keys():
                self.actual_history[h] = v
            else:
                self.actual_history[h] += v
        self.start_steps = self.start_steps + epochs
        # Make continue run false so that we run for all epochs in next segments!
        if self.continue_run:
            log.info("Setting continue run to False")
            self.continue_run = False

    def upscale(self, factor=2):
        if self.opt_kwargs['traj_opt_kwargs']['scheduler'] == 'exponential':
            self.opt_kwargs['traj_opt_kwargs']['learning_rate'] = self.opt_kwargs['traj_opt_kwargs']['learning_rate'] / factor
        self.base_traj.upscale(factor)
        default_multiopt_compile(self, reuse_recon_opt=True, **self.opt_kwargs)
         
    def multires_fit(self, epochs, start_steps=0, joint_fit=False, **kwargs):
        fit= self.trajectory_fit
        if joint_fit:
            fit = self.joint_fit
        if start_steps is not None:
            self.set_initializers(start_steps=start_steps)
        start_decim = self.base_traj.current_decim
        for i in range(int(np.log2(start_decim)) + 1):
            fit(
                epochs=epochs,
                shards=int(np.log2(start_decim)) + 1,
                index=i,
                **kwargs,
            )
            if self.base_traj.current_decim != 1:
                self.upscale(2)
        self.copy_history()
        return self.actual_history

    def segmented_fit(self, epochs, steps_per_epoch, segments=[3, 2, 1], start_steps=0, left_over=None, **kwargs):
        if start_steps is not None:
            self.set_initializers(start_steps=start_steps)
        start_decim = self.base_traj.current_decim
        if int(np.log2(self.base_traj.start_decim)) + 1 != np.sum(segments) or len(segments) != 3:
            raise ValueError('The total number of segments for fitting is not equal to ' +
                             str(int(np.log2(self.base_traj.start_decim)) + 1))
        if isinstance(steps_per_epoch, int):
            steps_per_epoch_traj, steps_per_epoch_recon = steps_per_epoch, steps_per_epoch
            log.info("Choosing same number of steps per epoch: {}".format(steps_per_epoch))
        else:
            steps_per_epoch_traj, steps_per_epoch_recon = steps_per_epoch
            log.info(
                "Choosing different number of steps per epoch: {} for trajectory and {} for reconstructor".format(
                    steps_per_epoch_traj, steps_per_epoch_recon
                )
            )
        global_fit_args = {
                'epochs': epochs,
                'shards': int(np.log2(self.base_traj.start_decim)) + 2,
                **kwargs,
            }
        ### THIS SEEMS USELESS
        ### if self.start_steps < epochs:
        ###     self.use_recon_adjoint = True
        ###     global_fit_args['steps_per_epoch'] = steps_per_epoch_traj
        ###     self.trajectory_fit(index=0, **global_fit_args)
        ### if self.start_steps < 2*epochs:
        ###     self.use_recon_adjoint = False
        ###     global_fit_args['steps_per_epoch'] = steps_per_epoch_recon
        ###     self.recon_net_fit(index=0, reuse_recon_opt=True, reuse_traj_opt=True, **global_fit_args)
        for i in np.arange(int(np.log2(start_decim)) + 1)[::-1]:
            fit_args = global_fit_args.copy()
            fit_args['epochs'] = epochs if not self.continue_run else left_over
            if i > int(np.log2(self.base_traj.start_decim)) - segments[0]:
                # Alternate descent betwen epochs
                fit_args['epochs'] = 1
                for j in range(epochs if not self.continue_run else left_over):
                    fit_args['steps_per_epoch'] = steps_per_epoch_traj
                    self.trajectory_fit(index=i+1, reuse_traj_opt=True, reuse_recon_opt=True, **fit_args)
                    fit_args['steps_per_epoch'] = steps_per_epoch_recon
                    self.recon_net_fit(index=i+1, reuse_traj_opt=True, reuse_recon_opt=True, **fit_args)
            elif i > int(np.log2(self.base_traj.start_decim)) - segments[0] - segments[1]:
                num_epoch_times_till_now = 2+segments[0]*2
                if not (self.start_steps//epochs - num_epoch_times_till_now) % 2:
                    # We dont learn trajectory again if we already learned it
                    fit_args['steps_per_epoch'] = steps_per_epoch_traj
                    self.trajectory_fit(index=i+1, reuse_traj_opt=True, reuse_recon_opt=True, **fit_args)
                fit_args['steps_per_epoch'] = steps_per_epoch_recon
                self.recon_net_fit(index=i+1, reuse_traj_opt=True, reuse_recon_opt=True, **fit_args)
            else:
                fit_args['epochs'] *= 2
                fit_args['steps_per_epoch'] = max(steps_per_epoch_traj, steps_per_epoch_recon)
                self.joint_fit(index=i+1, reuse_traj_opt=True, reuse_recon_opt=True, **fit_args)
            if self.base_traj.current_decim != 1:
                self.upscale(2)
        self.copy_history()
        return self.actual_history