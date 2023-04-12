import tensorflow as tf
from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d
from fastmri_recon.data.utils.crop import adjust_image_size

from .utils import virtual_coil_reconstruction


class ImagesBuilder(FastMRIDatasetBuilder):
    def __init__(self, crop_image_data=True, image_size=(320, 320), scale_factor=1e6, send_ch_images=False, **kwargs):
        self.image_size = image_size
        self.crop_image_data = crop_image_data
        self.scale_factor = scale_factor
        self.send_ch_images = send_ch_images
        super(ImagesBuilder, self).__init__(**kwargs)

    def get_images_from_kspace(self, kspace):
        images = ortho_ifft2d(kspace * self.scale_factor)
        images = adjust_image_size(images, self.image_size, multicoil=self.multicoil)
        return images

    def preprocessing(self, *inputs):
        if self.complex_image:
            images = inputs[0] * self.scale_factor
            images = adjust_image_size(images, self.image_size, multicoil=self.multicoil)
            image = virtual_coil_reconstruction(images)
            if not self.send_ch_images:
                images = image
        else:
            image, kspace = inputs
            image = image * self.scale_factor
            if self.send_ch_images:
                images = self.get_images_from_kspace(kspace)[0]
            else:
                images = adjust_image_size(image, self.image_size)
        if self.crop_image_data:
            return images, adjust_image_size(image, self.image_size)[..., None]
        else:
            output_shape = tf.shape(image)[1:][None, :]
            output_shape = tf.tile(output_shape, [tf.shape(image)[0], 1])
            return (images, output_shape), image[..., None]
