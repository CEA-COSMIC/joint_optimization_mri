from omegaconf import OmegaConf, DictConfig
import hydra
import operator
import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from jOpMRI.models.joint_model import AcqRecModel
from jOpMRI.utils.callbacks import get_callbacks
from jOpMRI.utils.evaluate import get_last_model, get_epoch_from_model_name
from jOpMRI.config import OASIS_DATA_DIR, CONFIG, DATA_DIR
from jOpMRI.data.utils import train_val_split_n_batch
from jOpMRI.models.utils.compile import default_multiopt_compile
from jOpMRI.data.tfrecords.vcc import read_data
from sparkling.utils.argparse import fix_n_check_params
from jOpMRI.data.calgary import KspaceReader
import shutil


def from_file_to_volume(filename, scale_facor=1, reverse=True, vol_shape=(176, 256, 256)):
    def _padND(img, bounding):
        vol = np.zeros(bounding, dtype=img.dtype)
        start = tuple(map(lambda a, da: a//2-da//2, bounding, img.shape))
        end = tuple(map(operator.add, start, img.shape))
        slices = tuple(map(slice, start, end))
        vol[slices] = img
        return vol
    volume = sitk.GetArrayFromImage(sitk.ReadImage(filename.numpy().decode('utf-8')))
    if reverse and min(volume.shape) != volume.shape[0]:
        volume = np.moveaxis(volume, -1, 0)
    volume = volume / np.mean(volume) * scale_facor.numpy()
    volume = volume.astype(np.complex64)
    volume = _padND(volume, vol_shape)
    return volume

def prep(data_set, scale_factor=1):
    data_set = data_set.map(lambda x: tf.py_function(
                from_file_to_volume,
                [x, scale_factor],
                [tf.complex64],
            )[0])
    data_set = data_set.map(lambda x: (x, tf.abs(x)))
    return data_set

@hydra.main(config_path='../configs', config_name='learner3D' + CONFIG)
def generic_learner(cfg: DictConfig) -> None:
    tf.keras.utils.set_random_seed(0)
    if cfg['outdir'] is not None:
        outdir = cfg['outdir']
        model_file = cfg['model_file']
        contin = cfg['continue']
        cfg = OmegaConf.load(os.path.join(outdir, '.hydra', 'config.yaml'))
        shutil.copy(os.path.join(outdir, '.hydra', 'config.yaml'), os.path.join(os.getcwd(), '.hydra', 'config.yaml'))
        cfg['continue'] =contin
        if model_file is None:
            cfg['model_file'] = os.path.join(outdir, 'checkpoints', get_last_model(outdir))
        else:
            cfg['model_file'] = os.path.join(outdir, 'checkpoints', model_file)
    traj_params = fix_n_check_params(cfg['trajectory'])
    if cfg['debug'] >= 50:
        tf.config.run_functions_eagerly(True)
    if cfg['data']['name'] == 'calgary':
        dataset = []
        if isinstance(cfg['data']['len_data'], int):
            len_data_train, len_data_val = cfg['data']['len_data'], None
        else:
            len_data_train, len_data_val = cfg['data']['len_data']
        for i, (path, len_data) in enumerate(zip([cfg['data']['train_path'], cfg['data']['val_path']], [len_data_train, len_data_val])):
            raw_dataset = read_data(
                os.path.join(DATA_DIR, path),
                n_samples=cfg['data']['n_samples'] if i == 0 else None,
                is2D=False,
                create_y_as_abs=False,
                normalize=True,
                im_size=traj_params['recon_params']['img_size'],
                cardinality=len_data,
                scale_factor=cfg['data']['scale_factor'],
            ) 
            dataset.append(train_val_split_n_batch(
                raw_dataset,
                1,
                batch_size=cfg['train']['batch_size'],
                shuffle=cfg['data']['shuffle'],
                prefetch=False,
            ))
        train_set, val_set = dataset
    elif cfg['data']['name'] == 'calgary_mc':
        dataset = KspaceReader(os.path.join(DATA_DIR, cfg['data']['train_path'])).preprocessed_ds
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
    if cfg['data']['name'] != 'calgary':
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
    if cfg['output']['profile']:
        profdir = os.path.join(os.getcwd(), 'tensorboard', 'profile')
        os.makedirs(profdir, exist_ok=True)
        tf.profiler.experimental.start(profdir)
    # Run Model once to make sure the optimizer picks up
    y = jOpModel(tf.cast(tf.zeros((1, 1, *cfg['trajectory']['recon_params']['img_size'])), tf.complex64))
    if cfg['debug'] >= 150:
        from jOpMRI.utils.debug import test_single_step
        test_single_step(jOpModel, train_set, cfg, single_data=False, accum_steps=cfg['train']['accum_steps'])
    current_epoch, left_over = None, None
    if cfg['model_file'] is not None:
        if cfg['continue']:
            current_epoch, left_over, jOpModel = get_epoch_from_model_name(cfg['model_file'], cfg, jOpModel)
        jOpModel.continue_load_weights(cfg['model_file'])
    # Callbacks
    callbacks = get_callbacks(
        outdir=os.getcwd(),
        **cfg['output']['callbacks'],
    )
    extra_kwargs = {}
    if cfg['train']['type'] in ('multi_resolution', 'fully_joint'):
        fit = jOpModel.multires_fit
        extra_kwargs['joint_fit'] = cfg['train']['type'] == 'fully_joint'
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
        start_steps=current_epoch,
        left_over=left_over,
        **extra_kwargs,
    )
    if cfg['output']['profile']:
        tf.profiler.experimental.stop()
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


