from omegaconf import DictConfig
import hydra

import os
import numpy as np
import tensorflow as tf

from jOpMRI.models.joint_model import AcqRecModel
from jOpMRI.utils.callbacks import get_callbacks
from jOpMRI.config import FASTMRI_DATA_DIR, CONFIG
from jOpMRI.data.utils import train_val_split_n_batch
from jOpMRI.data.tfrecords.vcc import read_data
from sparkling.utils.argparse import fix_n_check_params


@hydra.main(config_path='../configs', config_name='learner' + CONFIG)
def generic_learner(cfg: DictConfig) -> None:
    tf.keras.utils.set_random_seed(0)
    if cfg['debug'] >= 50:
        tf.config.run_functions_eagerly(True)
    data_set = read_data(
        path=os.path.join(FASTMRI_DATA_DIR, cfg['data']['train_path']),
        contrast=cfg['data']['contrast'],
        split_slices=True,
        slice_random=False,
        n_samples=cfg['data']['n_samples'],
        use_abs_image_input=cfg['data']['use_abs_image_input'],
    )
    if cfg['data']['len_data'] is None:
        if cfg['data']['contrast'] == 'AXT2':
            cfg['data']['len_data'] = cfg['data']['t2_len_data']
        else:
            cfg['data']['len_data'] = cfg['data']['t1_len_data']
    # Split to train and validation
    train_set, val_set = train_val_split_n_batch(
        data_set,
        cfg['data']['train_val_split'],
        batch_size=cfg['train']['batch_size'],
        shuffle=cfg['data']['shuffle'],
        len_data=cfg['data']['len_data']
    )
    if cfg['train']['num_steps_per_epoch'] == -1:
        cfg['train']['num_steps_per_epoch'] = cfg['data']['len_data']
    jOpModel = AcqRecModel(
        trajectory_kwargs=fix_n_check_params(cfg['trajectory']),
        acq_kwargs=cfg['acquisition'],
        recon_kwargs=cfg['reconstruction'],
        opt_kwargs=cfg['train']['optimizer'],
        nufft_implementation=cfg['train']['nufft_implementation'],
        batch_size=cfg['train']['batch_size'],
        cfg=cfg,
    )
    # Run Model once to make sure the optimizer picks up
    y = jOpModel(tf.cast(tf.zeros((1, *cfg['trajectory']['recon_params']['img_size'])), tf.complex64))
    if cfg['debug'] >= 150:
        from jOpMRI.utils.debug import test_single_step
        test_single_step(jOpModel, train_set, cfg)
    if cfg['model_file'] is not None:
        jOpModel.continue_load_weights(cfg['model_file'])
    # Callbacks
    callbacks = get_callbacks(
        outdir=os.getcwd(),
        run_name=cfg['run_name'],
        **cfg['output']['callbacks'],
    )
    extra_kwargs = {}
    if cfg['train']['type'] == 'multi_resolution':
        fit = jOpModel.multires_fit
    elif cfg['train']['type'] == 'segmented':
        fit = jOpModel.segmented_fit
        extra_kwargs['segments'] = cfg['train']['segments']
    elif cfg['train']['type'] == 'recon_net':
        fit = jOpModel.recon_net_fit
    else:
        raise ValueError('Cant find model training type : ' + cfg['train']['type'])
    history = fit(
        x=train_set.repeat(),
        steps_per_epoch=cfg['train']['num_steps_per_epoch'],
        epochs=cfg['train']['num_epochs'],
        validation_data=val_set.repeat(),
        validation_steps=cfg['train']['validation_steps'],
        callbacks=callbacks,
        verbose=1,
        **extra_kwargs,
    )
    traj = jOpModel.base_traj.trajectory.numpy()
    # Save results
    np.save(
        os.path.join(os.getcwd(), 'quick_results'), (traj, history),
    )
    if cfg['output']['get_grads']:
        from sparkling.utils.gradient import get_kspace_points_grads_n_slew
        from sparkling.utils.gradient import create_gradient_file
        grads, k0, slew = get_kspace_points_grads_n_slew(traj, cfg['trajectory'])
        create_gradient_file(
            grads,
            k0,
            os.path.join(os.getcwd(), 'gradients'),
            version=4.2,
            acq_params=cfg['trajectory'],
        )


if __name__ == '__main__':
    generic_learner()
