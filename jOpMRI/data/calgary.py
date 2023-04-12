import tensorflow as tf
from .utils import load_magnitude_from_dicom, load_phase_from_dicom
from pathlib import Path
import h5py
import numpy as np


def load_kspace_h5(path):
    with h5py.File(path.numpy().decode('utf-8'), 'r') as f:
        abs_kspace = np.moveaxis(f['kspace'][()], -1, 0)
        complex_kspace = abs_kspace[::2] + 1j * abs_kspace[1::2]
    return complex_kspace.astype('complex64')[None, :]


class KspaceReader:
    def __init__(self, data_path, scale_factor=1):
        self.path = Path(data_path)
        self.scale_factor = scale_factor
        if self.path.is_file():
            self._files = [self.path]
        else:
            self._files = sorted([ f for f in self.path.iterdir() if f.suffix == '.h5'])
        self.files_ds = tf.data.Dataset.from_tensor_slices(
                [str(f) for f in self._files],
        )
        self.raw_ds = self.files_ds.map(self.load)
        self.preprocessed_ds = self.raw_ds
        self.filtered_files = self._files

    def load(self, filename):
        print(filename)
        [kspaces] = tf.py_function(
            load_kspace_h5,
            [filename],
            [tf.complex64]
        )
        # Shape sizes : BatchSize, Nx, Ny, Nz, NCoils
        kspaces.set_shape((None, None, None, None, None))
        return kspaces
