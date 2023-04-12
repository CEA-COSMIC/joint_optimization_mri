import tensorflow as tf
from tensorflow.keras.models import Model

from tfkbnufft import kbnufft_adjoint
from jOpMRI.models.acquisition.utils import nufft


class Adjoint(Model):
    def __init__(self, interpob, img_size, dcomp=True, complex=False, multicoil=False, nufft_implementation='tensorflow-nufft', **kwargs):
        self.dcomp = dcomp
        self.complex = complex
        self.nufft_back = kbnufft_adjoint(interpob)
        self.multicoil = multicoil
        self.img_size = img_size
        self.nufft_implementation = nufft_implementation
        super(Adjoint, self).__init__(**kwargs)

    def call(self, input):
        if self.multicoil:
            if self.dcomp:
                kspace, trajectory, Smaps, (density_comp,) = input
                kspace = kspace * tf.cast(density_comp, kspace.dtype)
            else:
                kspace, trajectory, Smaps = input
        elif self.dcomp:
            kspace, trajectory, (density_comp, ) = input
            kspace = kspace * tf.cast(density_comp, kspace.dtype)
        else:
            kspace, trajectory, _ = input
        if self.nufft_implementation == 'tensorflow-nufft':
            recon = nufft(
                kspace,
                tf.transpose(trajectory),
                grid_shape=self.img_size,
                transform_type='type_1',
                fft_direction='backward'
            )
        else:
            recon = self.nufft_back(kspace, trajectory)
        if len(self.img_size) == 2:
            recon = recon[..., None]
        if self.multicoil:
            recon = tf.reduce_sum(recon * tf.math.conj(Smaps[..., None]), axis=1)
        if self.complex:
            return recon
        else:
            return tf.math.abs(recon)


class LearntGridDCp(Model):
    # FIXME WIP
    def __init__(self, interpob, complex=False, **kwargs):
        self.nufft_back = kbnufft_adjoint(interpob)
        # self.dcomp = dcomp
        self.complex = complex
        super(Adjoint, self).__init__(**kwargs)

    def call(self, input):
        if self.dcomp:
            kspace, trajectory, (density_comp_grid, ) = input
            # kspace_grid = grid_kspace(kspace)
            kspace = kspace * tf.cast(density_comp_grid, kspace.dtype)
        else:
            kspace, trajectory, _ = input
        recon = self.nufft_back(kspace, trajectory)[..., None]
        if self.complex:
            return recon
        else:
            return tf.math.abs(recon)
