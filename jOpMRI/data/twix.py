import numpy as np
import tensorflow as tf
from pathlib import Path
from mapvbvd import mapVBVD
from .utils import load_twix

class TwixReader:
    def __init__(self, data_path, scale_factor=1e6, is2D=True):
        self.path = Path(data_path)
        self.scale_factor = scale_factor
        self.is2D = is2D
        if self.path.is_file():
            self._files = [self.path]
        else:
            self._files = sorted(self.path.glob('*.dat'))
        self.files_ds = tf.data.Dataset.from_tensor_slices(
                [str(f) for f in self._files],
        )
        self.raw_ds = self.files_ds.map(self.load_raw)
        self.preprocessed_ds = self.raw_ds.map(self.preprocess)

    def load_raw(self, filename):
        data = tf.py_function(
            load_twix,
            [filename],
            [tf.complex64, tf.float32]
        )
        # [num_adc_samples, channels, num_shots, slices] = tf.shape(raw_kspace)
        data[0].set_shape((None, None, None, None))
        data[1] = tf.tile(data[1][None], [tf.shape(data[0])[-1], 1])
        if self.is2D:
            data[1].set_shape((None, 2))
        else:
            data[1].set_shape((None, 3))
        return data

    def preprocess(self, raw_kspace, shifts):
        # Data Shaping:
        kspace = tf.transpose(raw_kspace, perm=[3, 1, 2, 0])
        kspace = tf.reshape(kspace, (
            tf.shape(kspace)[0],
            tf.shape(kspace)[1],
            tf.shape(kspace)[2] * tf.shape(kspace)[3],
        )) * self.scale_factor
        return kspace, shifts
