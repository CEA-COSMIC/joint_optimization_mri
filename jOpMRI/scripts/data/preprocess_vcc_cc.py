#### THIS IS DEPRECATED, I have merged this into the main preprocess_vcc.py script

from omegaconf import DictConfig
import hydra
from tqdm import tqdm

import os, glob
import tensorflow as tf

from jOpMRI.config import FASTMRI_DATA_DIR
from jOpMRI.models.preprocess.coil_combination import VirtualCoilCombination
from jOpMRI.data.tfrecords.vcc import encode_vcc_example

from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
import tensorflow_io as tfio



@hydra.main(config_path='../configs', config_name='preprocess_vcc')
def preprocess_vcc(cfg: DictConfig) -> None:
    files = glob.glob("/volatile/Chaithya/JZ/Scratch/DATA/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel/Train/*.h5")
    coil_combiner = VirtualCoilCombination(return_true_image=True, scale_factor=cfg['data']['scale_factor'], kspace_input=True)
    for file in tqdm(files, total=len(files)):
        kspace = tfio.IOTensor.from_hdf5(file)('/kspace').to_tensor()
        kspace = tf.reshape(kspace, [*kspace.shape[:-1], kspace.shape[-1]//2, 2])
        kspace = tf.complex(kspace[..., 0], kspace[..., 1])
        kspace = tf.transpose(kspace) 
        coil_combined_data = coil_combiner(kspace)
        filename = os.path.join(cfg['outdir'], os.path.basename(filename)[:-3] + '.tfr')
        with tf.io.TFRecordWriter(str(filename)) as writer:
            writer.write(encode_vcc_example(coil_combined_data))


if __name__ == '__main__':
    preprocess_vcc()