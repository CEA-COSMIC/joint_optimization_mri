import os
import hydra

import numpy as np
import tensorflow as tf

from jOpMRI.data.utils import train_val_split_n_batch
from jOpMRI.utils.callbacks.utils import get_callbacks
from jOpMRI.data.spectrum import SpectrumBuilder
from jOpMRI.models.sampling_density import JointModel
from jOpMRI.config import FASTMRI_DATA_DIR

from fastmri_recon.models.training.compile import default_model_compile


@hydra.main(config_path='../configs', config_name='learn_loupe')
def density_loupe(cfg):
    if cfg['debug'] >= 50:
        tf.config.run_functions_eagerly(True)
    # Load Dataset
    data_set = SpectrumBuilder(
        path=os.path.join(FASTMRI_DATA_DIR, cfg['data']['train_path']),
        multicoil=cfg['data']['coils'] == 'multicoil',
        repeat=False,
        split_slices=True,
        contrast=cfg['data']['contrast'],
        return_images=True,
        shuffle=True,
        log_spectrum=False,
        prefetch=False,
        n_samples=cfg['data']['n_samples'],
        scale_factor=1e6,
    ).preprocessed_ds
    # Split to train and validation
    train_set, val_set = train_val_split_n_batch(
        data_set,
        cfg['data']['train_val_split'],
        batch_size=cfg['train']['batch_size']
    )
    jOpModel = JointModel(
        image_size=tuple(cfg['image_size']),
        multicoil=False, # We learn only in singlecoil case, even though we get data from multicoil (like brain)
        acq_kwargs=cfg['acquisition'],
        recon_kwargs=cfg['reconstruction'],
    )
    default_model_compile(jOpModel, lr=cfg['train']['optimizer']['lr'], loss=cfg['train']['optimizer']['loss'])
    # Callbacks
    callbacks = get_callbacks(
        outdir=os.getcwd(),
        run_name=cfg['run_name'],
        save_freq=cfg['output']['save_freq'],
    )
    jOpModel.fit(
        train_set.repeat(),
        steps_per_epoch=cfg['train']['num_steps_per_epoch'],
        initial_epoch=0,
        epochs=cfg['train']['num_epochs'],
        validation_data=val_set.repeat(),
        validation_steps=cfg['train']['validation_steps'],
        verbose=1,
        callbacks=callbacks,
    )
    density = jOpModel.acq_model.get_layer('prob_mask').mult[..., 0]
    density = density - tf.reduce_min(density)
    density = density / tf.reduce_sum(density)
    np.save(os.path.join(os.getcwd(), 'results.npy'), density)


if __name__ == '__main__':
    density_loupe()
