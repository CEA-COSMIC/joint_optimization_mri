import tensorflow as tf
from .utils import load_magnitude_from_dicom, load_phase_from_dicom
from pathlib import Path


class DicomReader:
    def __init__(self, data_path, image_size=(320, 320), scale_factor=10):
        self.path = Path(data_path)
        self.scale_factor = scale_factor
        self.image_size = image_size
        if self.path.is_file():
            self._files = [self.path]
        else:
            self._files = sorted(self.path.iterdir())
        self.files_ds = tf.data.Dataset.from_tensor_slices(
                [str(f) for f in self._files],
        )
        self.raw_ds = self.files_ds.map(self.load)
        self.preprocessed_ds = self.raw_ds.map(self.preproccess)

    def preproccess(self, mag_images, pha_images):
        mag_images = tf.image.flip_up_down(tf.image.resize(mag_images, self.image_size))
        pha_images = tf.image.flip_up_down(tf.image.resize(pha_images, self.image_size))
        ref_data = tf.cast(mag_images, tf.complex64) * tf.exp(1j * tf.cast(pha_images, tf.complex64))
        ref_data = tf.transpose(ref_data, perm=[2, 1, 0]) * self.scale_factor
        return ref_data[:, None], tf.abs(ref_data[:, None, ..., None])
        
    def load(self, filename):
        [mag_images] = tf.py_function(
            load_magnitude_from_dicom,
            [filename + '/MagImages'],
            [tf.float32]
        )
        [phase_images] = tf.py_function(
            load_phase_from_dicom,
            [filename + '/PhaImages'],
            [tf.float32]
        )
        mag_images.set_shape((None, None, None))
        phase_images.set_shape((None, None, None))
        return mag_images, phase_images
