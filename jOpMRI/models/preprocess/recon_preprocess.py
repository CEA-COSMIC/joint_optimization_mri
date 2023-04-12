import warnings

import tensorflow as tf
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.mri.dcomp_calc import calculate_density_compensator

from tf_fastmri_data.preprocessing_utils.extract_smaps import non_cartesian_extract_smaps
from tf_fastmri_data.preprocessing_utils.fourier.non_cartesian import nufft
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d

from fastmri_recon.data.utils.crop import adjust_image_size


class PreProcModel(tf.keras.models.Model):
    def __init__(self, traj, multicoil=True, crop_image_data=True, image_size=(320, 320), kspace_input=False,
                 dcomp=True, **kwargs):
        super(PreProcModel, self).__init__(**kwargs)
        self.image_size = image_size
        self.kspace_input = kspace_input
        self.nufft_ob = KbNufftModule(
            im_size=image_size,
            grid_size=None,
            norm='ortho',
        )
        self.traj = traj
        if multicoil and not dcomp:
            warnings.warn('Need Dcomp for Multicoil to estimate Smaps')
            dcomp = True
        self.dcomp = dcomp
        self.multicoil = multicoil
        self.crop_image_data = crop_image_data
        self.interpob = self.nufft_ob._extract_nufft_interpob()
        self.nufftob_forw = kbnufft_forward(self.interpob)
        self.nufftob_back = kbnufft_adjoint(self.interpob)
        if self.dcomp:
            self.density_comp = calculate_density_compensator(
                self.nufft_ob._extract_nufft_interpob(),
                self.nufftob_forw,
                self.nufftob_back,
                self.traj[0],
            )

    def call(self, inputs):
        if self.crop_image_data:
            orig_image_channels = inputs
        else:
            orig_image_channels, output_shape = inputs
        if self.kspace_input:
            # Obtain images
            orig_image_channels = adjust_image_size(
                ortho_ifft2d(orig_image_channels),
                self.image_size,
                multicoil=self.multicoil
            )
        batch_size = tf.shape(orig_image_channels)[0]
        traj = tf.repeat(self.traj, batch_size, axis=0)
        if not self.multicoil:
            orig_image_channels = orig_image_channels[:, None]
        nc_kspace = nufft(self.nufft_ob, orig_image_channels, traj, multicoil=self.multicoil)
        nc_kspaces_channeled = nc_kspace[..., None]
        orig_shape = tf.ones(batch_size, dtype=tf.int32) * self.image_size[-1]
        extra_args = (orig_shape, )
        if self.dcomp:
            dcomp = tf.ones(
                [batch_size, tf.shape(self.density_comp)[0]],
                dtype=self.density_comp.dtype,
            ) * self.density_comp[None, :]
            extra_args += (dcomp,)
        model_inputs = (nc_kspaces_channeled, traj)
        if self.multicoil:
            smaps = non_cartesian_extract_smaps(nc_kspace, traj, dcomp, self.nufftob_back, self.image_size)
            model_inputs += (smaps,)
        if not self.crop_image_data:
            model_inputs += (output_shape,)
        model_inputs += (extra_args,)
        return model_inputs
