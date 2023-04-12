from omegaconf import OmegaConf, DictConfig
import hydra

import os
import numpy as np
import tensorflow as tf

from jOpMRI.models.joint_model import AcqRecModel
from jOpMRI.utils.callbacks import get_callbacks
from jOpMRI.config import DATA_DIR, OASIS_DATA_DIR
from jOpMRI.data.tfrecords.vcc import read_data
from jOpMRI.data.utils import train_val_split_n_batch
from jOpMRI.utils.evaluate import get_last_model, get_epoch_from_model_name
from sparkling.utils.argparse import fix_n_check_params
import shutil


@hydra.main(config_path='../configs', config_name='continue')
def continue_3d(cfg: DictConfig) -> None:
    new_cfg = cfg
    cfg = OmegaConf.load(os.path.join(new_cfg['outdir'], '.hydra', 'config.yaml'))
    shutil.copy(os.path.join(new_cfg['outdir'], '.hydra', 'config.yaml'), os.path.join(os.getcwd(), '.hydra', 'config.yaml'))
    traj_params = fix_n_check_params(cfg['trajectory'])
    #os.chdir(new_cfg['outdir'])
    if new_cfg['debug'] >= 50:
        tf.config.run_functions_eagerly(True)
    if cfg['data']['name'] == 'calgary':
        dataset = read_data(
            os.path.join(DATA_DIR, cfg['data']['train_path']),
            n_samples=cfg['data']['n_samples'],
            is2D=False,
            create_y_as_abs=False,
            normalize=True,
            im_size=traj_params['recon_params']['img_size'],
            cardinality=cfg['data']['len_data'],
            scale_factor=cfg['data']['scale_factor'],
        ) 
    else:
        filtered_files = []
        for contrast in cfg['data']['contrast']:
            filtered_files += list(np.load(os.path.join(
                OASIS_DATA_DIR,
                cfg['data']['train_path'],
                contrast
            ) + '_ff.npy'))
        filtered_files = [
            os.path.join(OASIS_DATA_DIR, cfg['data']['train_path'], f)
            for f in filtered_files
        ]
        files_ds = tf.data.Dataset.from_tensor_slices(filtered_files)
        dataset = prep(files_ds, scale_factor=cfg['data']['scale_factor'])
    if cfg['data']['n_samples'] is not None:
        dataset = dataset.take(cfg['data']['n_samples'])
    if cfg['train']['num_steps_per_epoch'] == -1:
        cfg['train']['num_steps_per_epoch'] = cfg['data']['len_data']
    train_set, val_set = train_val_split_n_batch(
        dataset,
        cfg['data']['train_val_split'],
        batch_size=cfg['train']['batch_size'],
        shuffle=cfg['data']['shuffle'],
        prefetch=False,
    )
    jOpModel = AcqRecModel(
        trajectory_kwargs=traj_params,
        acq_kwargs=cfg['acquisition'],
        recon_kwargs=cfg['reconstruction'],
        opt_kwargs=cfg['train']['optimizer'],
        nufft_implementation=cfg['train']['nufft_implementation'],
        batch_size=cfg['train']['batch_size'],
        accum_steps=cfg['train']['accum_steps'],
        cfg=cfg,
    )
    if new_cfg['model_file'] is None:
        # Get the latest model
        new_cfg['model_file'] = get_last_model(new_cfg['outdir'])
    y = jOpModel(tf.cast(tf.zeros((1, 1, *cfg['trajectory']['recon_params']['img_size'])), tf.complex64))
    current_epoch, left_over, jOpModel = get_epoch_from_model_name(new_cfg['model_file'], cfg, jOpModel)
    jOpModel.continue_load_weights(os.path.join(new_cfg['outdir'], 'checkpoints', new_cfg['model_file']))
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
    elif cfg['train']['type'] == 'full_joint':
        fit = jOpModel.multires_joint_fit
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
        start_steps=current_epoch,
        left_over=left_over,
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
    continue_3d()