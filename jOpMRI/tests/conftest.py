import os

import numpy as np
import tensorflow as tf

import pytest

from tf_fastmri_data.test_utils import create_full_fastmri_tmp_dataset, ktraj_function

from sparkling.utils.trajectory import init_shots
from sparkling.utils.gradient import create_gradient_file, get_kspace_points_grads_n_slew

from tf_fastmri_data.test_utils import key_map
from hydra import initialize, compose
from jOpMRI.scripts.data.preprocess_vcc import preprocess_vcc


fake_params = {
        'recon_params': {
            'FOV': (0.23, 0.23),
            'img_size': (320, 320),
        },
        'scan_consts': {
            'gyromagnetic_constant': 42.576e3,
            'gradient_raster_time': 0.010,
        },
        'traj_params': {
            'oversampling_factor': 5,
            'initialization': 'RadialCO',
            'num_shots': 16,
            'num_samples_per_shot': 513,
            'dimension': 2,
        }
    }

@pytest.fixture(scope='session', autouse=False)
def ktraj():
    return ktraj_function

@pytest.fixture(scope="session", autouse=False)
def create_full_fastmri_test_tmp_dataset(tmpdir_factory):
    tf.random.set_seed(0)
    return create_full_fastmri_tmp_dataset(tmpdir_factory)

@pytest.fixture(scope="session", autouse=False)
def set_outdir(tmpdir_factory):
    os.chdir(str(tmpdir_factory.mktemp('output')))
    return


@pytest.fixture(scope="session", autouse=False)
def create_tmp_trajectory(tmpdir_factory):
    np.random.seed(0)
    shots, _ = init_shots(
        num_shots=fake_params['traj_params']['num_shots'],
        num_samples_per_shot=fake_params['traj_params']['num_samples_per_shot'],
        dimension=fake_params['traj_params']['dimension'],
        initialization=fake_params['traj_params']['initialization'],
        perturbation_factor=0.1,
    )
    grads, k0, slew = get_kspace_points_grads_n_slew(
        shots=shots,
        params=fake_params,
        check=False,
    )
    traj_dir = tmpdir_factory.mktemp(
        "trajectories",
        numbered=False,
    )
    create_gradient_file(
        grads,
        k0,
        os.path.join(traj_dir, fake_params['traj_params']['initialization']),
        acq_params=fake_params
    )
    return {'traj_dir': str(traj_dir)}

@pytest.fixture(params=(create_full_fastmri_test_tmp_dataset,), scope="session", autouse=False)
def create_tfrecords_complex(create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset[key_map[True]['train']]
    with initialize(config_path=os.path.join('__file__', '..', '..', 'scripts', 'configs')):
        cfg = compose(
            config_name='preprocess_vcc',
            overrides=[
                'data.train_path=' + path,
                'outdir=' + os.path.join(os.path.dirname(path[:-1]), 'tfrecords'),
                'data.scale_factor=1',
                'data.contrast=null',
            ]
        )
        preprocess_vcc(cfg)
    create_full_fastmri_test_tmp_dataset['tfrecords'] = os.path.join(os.path.dirname(path[:-1]), 'tfrecords')
    return create_full_fastmri_test_tmp_dataset
