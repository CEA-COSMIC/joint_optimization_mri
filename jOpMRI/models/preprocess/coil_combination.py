import tensorflow as tf
from jOpMRI.data.utils import virtual_coil_reconstruction

from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d

from fastmri_recon.data.utils.crop import adjust_image_size
from tensorflow.python.ops.signal.fft_ops import ifft2d, ifftshift


class VirtualCoilCombination(tf.keras.models.Model):
    def __init__(self, return_true_image=False, scale_factor=1e6, image_size=(320, 320), 
                 kspace_input=False, data_name='fastmri_brain', **kwargs):
        self.return_true_image = return_true_image
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.data_name = data_name
        self.kspace_input = kspace_input
        super(VirtualCoilCombination, self).__init__(**kwargs)

    def call(self, inputs):
        if self.kspace_input:
            kspace = inputs
            kspace = kspace * self.scale_factor
        else:
            true_image, kspace = inputs
            true_image = true_image * self.scale_factor
            kspace = kspace[0] * self.scale_factor
        if self.data_name == 'calgary':
            # This dataset is organized as Nch x X, Ky, Kz
            images = ifft2d(ifftshift(kspace, axes=[-2, -1]))
            if tf.shape(images)[-1] != self.image_size[-1]:
                D = (tf.shape(images)[-1] - self.image_size[-1])//2
                images = images[..., D:-D]
        else:
            images = ortho_ifft2d(kspace)
            images = adjust_image_size(images, self.image_size, multicoil=True)
        if not self.kspace_input:
            true_image = adjust_image_size(true_image[..., 0], self.image_size)[..., None]
        complex_image = virtual_coil_reconstruction(images)
        if self.return_true_image:
            return complex_image, true_image
        else:
            return complex_image, tf.abs(complex_image)