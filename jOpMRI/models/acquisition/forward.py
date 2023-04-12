import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

from tfkbnufft import kbnufft_forward
from jOpMRI.models.acquisition.utils import nufft

from sparkling.utils.trajectory import get_time_vector


class BasicForward(Model):
    def __init__(self, interpob, nufft_implementation='tensorflow-nufft', **kwargs):
        self.nufft_forw = kbnufft_forward(interpob)
        self.nufft_implementation = nufft_implementation
        super(BasicForward, self).__init__(**kwargs)

    def call(self, input):
        image, trajectory = input
        if self.nufft_implementation == 'tensorflow-nufft':
            kspace = nufft(tf.cast(image, tf.complex64), tf.transpose(trajectory), transform_type='type_2', fft_direction='forward')
        else:
            kspace = self.nufft_forw(tf.cast(image, tf.complex64), trajectory)
        return kspace


class ConstantB0T2(BasicForward):
    def __init__(self, TE=1, Nc=32, Ns=513, osf=5, gradient_raster_time=0.01, T2=80e-3, B0=25, **kwargs):
        super(ConstantB0T2, self).__init__(**kwargs)
        self.time = tf.tile(get_time_vector(
            num_samples=Ns*osf,
            k_TE=TE,
            echo_time_ms=0,
            dwell_time=gradient_raster_time/osf,
        ), [Nc])
        self.T2 = T2
        self.B0 = B0
        if not(isinstance(self.B0, int) and isinstance(self.T2, float)) and (len(self.T2) == 2 and len(self.B0) == 2):
            self.random = True
        else:
            self.random = False
            self.kspace_multiplier = tf.exp(
                tf.cast(-self.time / self.T2, tf.complex64) +
                1j * 2 * np.pi * self.B0 * tf.cast(self.time, tf.complex64)
            )

    def call(self, input):
        kspace = super(ConstantB0T2, self).call(input)
        if self.random:
            kspace_multiplier = tf.exp(
                tf.cast(-self.time / np.random.uniform(*self.T2), tf.complex64) +
                1j * 2 * np.pi * np.random.uniform(*self.B0) * tf.cast(self.time, tf.complex64)
            )
        else:
            kspace_multiplier = self.kspace_multiplier
        kspace = kspace * kspace_multiplier[None, None]
        return kspace