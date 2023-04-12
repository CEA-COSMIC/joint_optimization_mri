from omegaconf import OmegaConf, DictConfig
import hydra
from tqdm import tqdm
import warnings

import os
import tensorflow as tf
import pickle as pkl

from jOpMRI.models.joint_model import AcqRecModel
from jOpMRI.config import DATA_DIR
from jOpMRI.data.images import ImagesBuilder
from jOpMRI.data.tfrecords.vcc import read_data
from jOpMRI.data.twix import TwixReader
from jOpMRI.data.dicom import DicomReader
from jOpMRI.data.utils import shuffle_batch_prefetch
from jOpMRI.utils.evaluate import METRIC_FUNCS, get_last_model

from sparkling.utils.argparse import fix_n_check_params
from fastmri_recon.evaluate.metrics.np_metrics import Metrics

def remove_refine_smaps(cfg):
    if 'refine_smaps' in cfg['reconstruction']['params'].keys():
        # For now, we dont use refine_smaps for prospective
        cfg['reconstruction']['params']['refine_smaps'] = False
    return cfg


@hydra.main(config_path='../configs', config_name='evaluate')
def evaluate_recon(cfg: DictConfig) -> None:
    tf.keras.utils.set_random_seed(0)
    run_cfg = OmegaConf.load(os.path.join(cfg['outdir'], '.hydra', 'config.yaml'))
    traj_params = fix_n_check_params(run_cfg['trajectory'])
    if cfg['debug'] >= 50:
        tf.config.run_functions_eagerly(True)
    if cfg['data']['n_samples'] is None:
        cfg['data']['n_samples'] = 512
    if cfg['base_data_type'] == 'prospective':
        cfg['multicoil'] = True  # Scanner data is always multicoil
        run_cfg = remove_refine_smaps(run_cfg)
        val_set = shuffle_batch_prefetch(TwixReader(cfg['datadir']).preprocessed_ds, unbatch=True, batch_size=cfg['batch_size'])
        X = next(val_set.unbatch().batch(1).as_numpy_iterator())
    elif cfg['base_data_type'] == 'dicom':
        cfg['multicoil'] = False  # We dont support multicoil datset yet
        val_set = shuffle_batch_prefetch(DicomReader(cfg['datadir']).preprocessed_ds, unbatch=True, batch_size=cfg['batch_size'])
        X, y_true = next(val_set.as_numpy_iterator())
    elif cfg['base_data_type'] == 'fastMRI':
        if cfg['multicoil']:
            val_set = ImagesBuilder(
                path=os.path.join(DATA_DIR, cfg['data']['val_path']),
                contrast=run_cfg['data']['contrast'],
                multicoil=True,
                split_slices=True,
                slice_random=False,
                n_samples=cfg['data']['n_samples'],
                complex_image=False,
                shuffle=False,
                send_ch_images=True,
                repeat=False,
            ).preprocessed_ds
            val_set = shuffle_batch_prefetch(val_set, shuffle=cfg['data']['n_samples']*5, batch_size=cfg['batch_size'])
            run_cfg = remove_refine_smaps(run_cfg)
        else:
            val_set=read_data(
                path=os.path.join(DATA_DIR, cfg['data']['val_path']),
                contrast=run_cfg['data']['contrast'],
                split_slices=True,
                slice_random=False,
                n_samples=cfg['data']['n_samples'],
                use_abs_image_input=run_cfg['data']['use_abs_image_input'],
            )
            val_set = shuffle_batch_prefetch(val_set, shuffle=cfg['data']['n_samples']*5, batch_size=cfg['batch_size'])
        X, y_true = next(val_set.unbatch().batch(1).as_numpy_iterator())
    else:
        val_set = read_data(
            os.path.join(DATA_DIR, cfg['data']['test_path']),
            n_samples=cfg['data']['n_samples'],
            is2D=False,
            create_y_as_abs=False,
            normalize=True,
            im_size=traj_params['recon_params']['img_size'],
            cardinality=run_cfg['data']['len_data'][-1],
            scale_factor=run_cfg['data']['scale_factor'],
        ).batch(1)
        X, y_true = next(val_set.as_numpy_iterator())

    # We need this for better model in prospective, TODO, see what are better ways to do this
    run_cfg['acquisition']['type'] = 'basic'
    jOpModel=AcqRecModel(
        trajectory_kwargs=traj_params,
        acq_kwargs=run_cfg['acquisition'],
        recon_kwargs=run_cfg['reconstruction'],
        opt_kwargs=run_cfg['train']['optimizer'],
        multicoil=cfg['multicoil'],
        prospective=cfg['base_data_type'] == 'prospective',
        nufft_implementation=run_cfg['train']['nufft_implementation'] if 'nufft_implementation' in run_cfg['train'].keys() else 'tensorflow-nufft',
        cfg=run_cfg,
    )
    if cfg['model_file'] is None:
        # Get the latest model
        cfg['model_file'] = get_last_model(cfg['outdir'])
    model_filename = os.path.join(cfg['outdir'], 'checkpoints', cfg['model_file'])
    jOpModel.upscale_to_match_model_file(model_filename, remove_last_point=cfg['base_data_type'] == 'prospective')
    y = jOpModel(X)
    jOpModel.continue_load_weights(model_filename, load_opt=False)
    if jOpModel.base_traj.current_decim != 1:
        warnings.warn('This seems to be a model that is not trained fully!')
    metrics = Metrics(METRIC_FUNCS, os.path.join(os.getcwd(), 'metrics.csv'))
    saved=False
    len_data = cfg['data']['n_samples']
    if len(val_set) != -1 and len(val_set) < len_data:
        len_data = len(val_set)
    i = 0
    for x_batch, y_true_batch in tqdm(val_set.as_numpy_iterator(), total=len_data):
        if cfg['base_data_type'] == 'prospective':
            x_batch = (x_batch, y_true_batch) # y_true_batch is shifts, needed as input
        y_pred_batch = jOpModel.predict(x_batch)
        if cfg['debug'] >= 2 and not saved:
            pkl.dump((y_true_batch, y_pred_batch), open('result_batch_' + str(i) + '.pkl', 'wb'))
            i += 1
            if i > cfg['num_save']:
                saved = True
        if cfg['base_data_type'] != 'prospective':
            for y_pred, y_true in zip(y_pred_batch, y_true_batch):
                if  cfg['base_data_type'] == 'calgary':
                    metrics.push(y_true[0], y_pred[0])
                else:
                    metrics.push(y_true[0, ..., 0], y_pred[0, ..., 0])
    metrics.to_csv()


if __name__ == '__main__':
    evaluate_recon()
