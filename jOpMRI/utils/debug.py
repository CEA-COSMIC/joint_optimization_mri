import tensorflow as tf
import numpy as np
from .debug_imports import *
from jOpMRI.models.acquisition.utils import get_density_compensators
from jOpMRI.models.utils.compile import default_multiopt_compile
import tensorflow_nufft as tfft
from sparkling.utils.trajectory import init_shots


def test_single_step(jOpModel, train_set, cfg, eager=False, single_data=True, custom_fit=True, accum_steps=1):
    iterator = train_set.as_numpy_iterator()
    if eager:
        tf.config.run_functions_eagerly(True)
        jOpModel.run_eagerly = eager
    losses = []
    start_decim = jOpModel.base_traj.current_decim
    if cfg['train']['type'] == 'recon_net':
        if not jOpModel.traj_loaded and cfg['trajectory']['traj_init'] is not None:
            jOpModel.init_trajectory(cfg['trajectory']['traj_init'])
        jOpModel.base_traj.trainable = False
        default_multiopt_compile(jOpModel, loss='compound_l1_l2', reuse_traj_opt=True, reuse_recon_opt=True)
    if custom_fit:
        for log_decim in range(int(np.log2(start_decim)) + 1):
            accum_ctr = 0
            accum_grads = [tf.zeros_like(var) for var in jOpModel.trainable_variables]
            for i in range(100):
                if (i == 0 and log_decim == 0) or not single_data:
                    input = next(iterator)
                    image = input[0]
                    ref_image = input[1]
                with tf.GradientTape() as tape:
                    accum_ctr += 1
                    y = jOpModel(image)
                    loss = tf.reduce_mean(jOpModel.loss(ref_image, y)) / accum_steps
                    losses.append(loss)
                    print(loss)
                grads = tape.gradient(loss, jOpModel.trainable_variables)
                if accum_ctr % accum_steps == 0:
                    if accum_steps == 1:
                        accum_grads = grads
                    jOpModel.optimizer.apply_gradients(zip(accum_grads, jOpModel.trainable_variables))
                    accum_grads = grads
                else:
                    for i, g in enumerate(grads):
                        accum_grads[i] += g
            y = jOpModel(image)
            loss = tf.reduce_mean(jOpModel.loss(ref_image, y)) / accum_steps
            print("Loss before upscale")
            print(loss)
            jOpModel.upscale(2)
            y = jOpModel(image)
            loss = tf.reduce_mean(jOpModel.loss(ref_image, y)) / accum_steps
            print("Loss after upscale")
            print(loss)
            opt = jOpModel.optimizer
    else:
        jOpModel.fit(train_set, epochs=100, steps_per_epoch=1)
    return losses, jOpModel


class TrajModel(tf.keras.Model):
    def __init__(self, shape=(2, 32*513)):
        super(TrajModel, self).__init__()
        self.shape = shape
        traj_params = {'dimension': 2, 'num_shots': 32, 'initialization': 'RadialIO', 'num_samples_per_shot': 513,
                       'oversampling_factor': 5, 'perturbation_factor': 0.75, 'num_revolutions': 2}
        self.init, _ = init_shots(**traj_params)
        self.traj = self.add_weight(
            shape=self.shape,
            dtype=tf.float32,
        )
        self.traj.assign(tf.cast(tf.transpose(2 * np.pi * np.pi * self.init.reshape(-1, 2)), tf.float32))

    def call(self, image):
        density_comp = get_density_compensators(self.traj)
        kspace = tfft.nufft(image, tf.transpose(self.traj),
                            transform_type='type_2', fft_direction='forward')
        kspace = kspace * tf.cast(density_comp, kspace.dtype)
        recon = tfft.nufft(
            kspace,
            tf.transpose(self.traj),
            grid_shape=image.shape[-2:],
            transform_type='type_1',
            fft_direction='backward'
        )
        return tf.abs(recon[..., None])

def test_single_step_basic(train_set, eager=False, jOpModel=None):
    if jOpModel is None:
        jOpModel = TrajModel()
    input = next(train_set.as_numpy_iterator())
    image = input[0]
    ref_image = input[1]
    if eager:
        tf.config.run_functions_eagerly(True)
        jOpModel.run_eagerly = eager
    try:
        opt = jOpModel.optimizer
    except:
        opt = tf.optimizer.Adam(lr=1e-3)
    losses = []
    start_decim = 1
    for i in range(1000):
        print(i)
        with tf.GradientTape() as tape:
            y = jOpModel(image)
            loss = tf.reduce_mean(1-tf.image.ssim_multiscale(ref_image, y, max_val=tf.reduce_max(ref_image)))
            if tf.math.reduce_any(tf.math.is_nan(tf.abs(y))):
                print("Got NaN!")
                exit(0)
            #loss = tf.reduce_mean(tf.abs(tf.abs(y) - image))
            #loss2 = loss + jOpModel.layers[1].losses[0]
            #loss3 = jOpModel.layers[1].losses[0]
            #losses.append(loss)
            print("Max value of Recon : " + str(np.max(y)))
            print(loss)
        #grads = tape.gradient(loss, jOpModel.trainable_variables)
        #grads2 = tape.gradient(loss2, jOpModel.trainable_variables)
        #grads3 = tape.gradient(loss3, jOpModel.trainable_variables)
        #opt.apply_gradients(zip(grads, jOpModel.trainable_variables))
    opt = jOpModel.optimizer
    exit(0)
    return losses