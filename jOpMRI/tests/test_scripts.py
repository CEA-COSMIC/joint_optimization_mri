import pytest
import os
import tensorflow as tf

from jOpMRI.scripts.training.learner import generic_learner
from jOpMRI.scripts.training.learn_loupe_density import density_loupe
from jOpMRI.scripts.spectrum_density import get_sampling_densities

from hydra import initialize, compose
from tf_fastmri_data.test_utils import key_map


@pytest.mark.parametrize('reconstruction', ['adjoint', 'ncpdnet_small_test', 'unet_small_test'])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('acquisition', ['nufft', 'constant_t2b0'])
@pytest.mark.parametrize('train_type', ['multi_resolution', 'segmented'])
def test_trajectory(reconstruction, batch_size, acquisition, train_type, create_tfrecords_complex, set_outdir):
    tf.keras.utils.set_random_seed(0)
    train_path = create_tfrecords_complex['tfrecords']
    if train_type == 'segmented' and reconstruction == 'adjoint':
        pytest.skip('Bad use case, skipping')
    if acquisition == 'constant_t2b0' and reconstruction != 'adjoint':
        pytest.skip('Bad use case, skipping')
    with initialize(config_path=os.path.join('__file__', '..', '..', 'scripts', 'configs')):
        cfg = compose(
            config_name='learner_local',
            overrides=[
                'data.train_path='+train_path,
                'data.contrast=null',
                'data.n_samples=1',
                'data.shuffle=1',
                'data.len_data=1',
                'data.train_val_split=0.5',
                'reconstruction=' + reconstruction,
                'acquisition=' + acquisition,
                'train.num_epochs=1',
                'train.num_steps_per_epoch=1',
                'train.batch_size=' + str(batch_size),
                'output.save_freq=2',
                'trajectory.algo_params.start_decim=' + str(4 if train_type == 'segmented' else 2),
                'train.type=' + train_type,
                'train.segments=[1, 1, 1]',
                'data.contrast=null',
                'trajectory.algo_params.max_proj_iter=10',
            ]
        )
        generic_learner(cfg)


def test_spectrum_density(create_full_fastmri_test_tmp_dataset, set_outdir):
    train_path = create_full_fastmri_test_tmp_dataset[key_map[True]['train']]
    val_path = create_full_fastmri_test_tmp_dataset[key_map[True]['val']]
    with initialize(config_path=os.path.join('__file__', '..', '..', 'scripts', 'configs')):
        cfg = compose(
            config_name='spectrum_density',
            overrides=[
                'data.train_path=' + train_path,
                'data.val_path=' + val_path,
                'data.contrast=null'
            ]
        )
        get_sampling_densities(cfg)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_density_learner(batch_size, create_full_fastmri_test_tmp_dataset, set_outdir):
    train_path = create_full_fastmri_test_tmp_dataset[key_map[True]['train']]
    val_path = create_full_fastmri_test_tmp_dataset[key_map[True]['val']]
    with initialize(config_path=os.path.join('__file__', '..', '..', 'scripts', 'configs')):
        cfg = compose(
            config_name='learn_loupe',
            overrides=[
                'data.train_path=' + train_path,
                'data.val_path=' + val_path,
                'train.num_epochs=2',
                'train.num_steps_per_epoch=1',
                'train.batch_size=' + str(batch_size),
                'data.contrast=null'
            ]
        )
        density_loupe(cfg)


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('reconstruction', ['ncpdnet_small_test', 'unet_small_test'])
def test_learn_recon(batch_size, reconstruction, create_tfrecords_complex, create_tmp_trajectory, set_outdir):
    train_path = create_tfrecords_complex['tfrecords']
    traj_path = os.path.join(create_tmp_trajectory['traj_dir'], 'RadialCO.bin')
    with initialize(config_path=os.path.join('__file__', '..', '..', 'scripts', 'configs')):
        cfg = compose(
            config_name='learner',
            overrides=[
                'data.train_path=' + train_path,
                'trajectory.traj_init=' + traj_path,
                'data.contrast=null',
                'data.n_samples=1',
                'data.shuffle=1',
                'data.len_data=1',
                'data.train_val_split=0.5',
                'train.num_epochs=2',
                'reconstruction='+reconstruction,
                'train.num_steps_per_epoch=1',
                'output.save_freq=2',
                'trajectory.algo_params.start_decim=1',
                'trajectory.traj_params.num_shots=16',
                'train.type=recon_net',
                'train.batch_size=' + str(batch_size),
            ]
        )
        generic_learner(cfg)
