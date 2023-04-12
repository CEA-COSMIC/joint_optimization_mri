from omegaconf import DictConfig
import hydra
from tqdm import tqdm

import os
import tensorflow as tf

from jOpMRI.config import DATA_DIR
from jOpMRI.models.preprocess.coil_combination import VirtualCoilCombination
from jOpMRI.data.tfrecords.vcc import encode_vcc_example

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from jOpMRI.data.calgary import KspaceReader
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name='preprocess_vcc')
def preprocess_vcc(cfg: DictConfig) -> None:
    if cfg['debug'] >= 50:
        tf.config.run_functions_eagerly(True)
    if cfg['data']['name'] == 'calgary':
        dataset = KspaceReader(os.path.join(DATA_DIR, cfg['data']['train_path']))
    else:
        dataset = FastMRIDatasetBuilder(
            path=os.path.join(DATA_DIR, cfg['data']['train_path']),
            multicoil=True,
            repeat=False,
            contrast=cfg['data']['contrast'],
            shuffle=False,
            prefetch=True,
        )
    filenames = dataset.filtered_files
    data_points = dataset.preprocessed_ds
    coil_combiner = VirtualCoilCombination(
        return_true_image=cfg['data']['name'] != 'calgary',
        scale_factor=cfg['data']['scale_factor'],
        kspace_input=cfg['data']['name'] == 'calgary',
        image_size=cfg['image_size'],
        data_name=cfg['data']['name'],
    )
    for filename, data_point in tqdm(zip(filenames, data_points.as_numpy_iterator()), total=len(filenames)):
        coil_combined_data = coil_combiner(data_point)
        filename = os.path.join(os.getcwd(), os.path.basename(filename)[:-3] + '.tfr')
        with tf.io.TFRecordWriter(str(filename)) as writer:
            writer.write(encode_vcc_example(coil_combined_data))

if __name__ == '__main__':
    preprocess_vcc()