from omegaconf import OmegaConf, DictConfig
import hydra

import os
import numpy as np
import tensorflow as tf

from jOpMRI.models.joint_model import AcqRecModel
from jOpMRI.utils.callbacks import get_callbacks
from jOpMRI.config import FASTMRI_DATA_DIR
from jOpMRI.data.tfrecords.vcc import read_data
from jOpMRI.data.utils import train_val_split_n_batch
from jOpMRI.utils.evaluate import get_last_model, get_epoch_from_model_name
from sparkling.utils.argparse import fix_n_check_params


@hydra.main(config_path='../configs', config_name='continue')
def evaluate_recon(cfg: DictConfig) -> None:
    run_cfg = OmegaConf.load(os.path.join(cfg['outdir'], '.hydra', 'config.yaml'))
    os.chdir(cfg['outdir'])
    if cfg['debug'] >= 50:
        tf.config.run_functions_eagerly(True)
    data_set = read_data(
        path=os.path.join(FASTMRI_DATA_DIR, run_cfg['data']['train_path']),
        contrast=run_cfg['data']['contrast'],
        split_slices=True,
        slice_random=False,
        n_samples=run_cfg['data']['n_samples'],
    )
    if run_cfg['data']['len_data'] is None:
        if run_cfg['data']['contrast'] == 'AXT2':
            run_cfg['data']['len_data'] = run_cfg['data']['t2_len_data']
        else:
            run_cfg['data']['len_data'] = run_cfg['data']['t1_len_data']
    # Split to train and validation
    train_set, val_set = train_val_split_n_batch(
        data_set,
        run_cfg['data']['train_val_split'],
        batch_size=run_cfg['train']['batch_size'],
        shuffle=run_cfg['data']['shuffle'],
        len_data=run_cfg['data']['len_data']
    )
    # Callbacks
    callbacks = get_callbacks(
        outdir=os.getcwd(),
        run_name=run_cfg['run_name'],
        **cfg['output']['callbacks'],
    )
    jOpModel = AcqRecModel(
        trajectory_kwargs=fix_n_check_params(run_cfg['trajectory']),
        acq_kwargs=run_cfg['acquisition'],
        recon_kwargs=run_cfg['reconstruction'],
        opt_kwargs=run_cfg['train']['optimizer'],
        batch_size=run_cfg['train']['batch_size'],
    )
    if cfg['model_file'] is None:
        # Get the latest model
        cfg['model_file'] = get_last_model(cfg['outdir'])
    current_epoch, left_over, type = get_epoch_from_model_name(cfg['model_file'], run_cfg)
    y = jOpModel(next(train_set.as_numpy_iterator())[0])
    jOpModel.continue_load_weights(os.path.join(cfg['outdir'], 'checkpoints', cfg['model_file']))
    jOpModel.set_initializers(current_epoch)
    if run_cfg['train']['type'] == 'multi_resolution':
        fit = jOpModel.multires_fit
    elif run_cfg['train']['type'] == 'segmented':
        fit = jOpModel.segmented_fit
    elif run_cfg['train']['type'] == 'recon_net':
        fit = jOpModel.recon_net_fit
    else:
        raise ValueError('Cant find model training type : ' + cfg['train']['type'])
    history = fit(
        x=train_set.repeat(),
        steps_per_epoch=run_cfg['train']['num_steps_per_epoch'],
        epochs=run_cfg['train']['num_epochs'],
        validation_data=val_set.repeat(),
        validation_steps=run_cfg['train']['validation_steps'],
        callbacks=callbacks,
        verbose=1,
        segments=run_cfg['train']['segments'],
        start_steps=None,
    )
    traj = jOpModel.base_traj.trajectory.numpy()
    # Save results
    np.save(
        os.path.join(os.getcwd(), 'quick_results'), (traj, history),
    )
    if run_cfg['output']['get_grads']:
        from sparkling.utils.gradient import get_kspace_points_grads_n_slew
        from sparkling.utils.gradient import create_gradient_file
        grads, k0, slew = get_kspace_points_grads_n_slew(traj, run_cfg['trajectory'])
        create_gradient_file(
            grads,
            k0,
            os.path.join(os.getcwd(), 'gradients'),
            version=4.2,
            acq_params=run_cfg['trajectory'],
        )

if __name__ == '__main__':
    evaluate_recon()