import tensorflow as tf
from tf_fastmri_data.dataset_builder import FastMRIDatasetBuilder
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_fft2d
from fastmri_recon.data.utils.crop import adjust_image_size


class SpectrumBuilder(FastMRIDatasetBuilder):
    def __init__(self, log_spectrum=True, return_images=False, target_shape=(320, 320), scale_factor=1, eps=1e-20, **kwargs):
        self.return_images = return_images
        self.log_spectrum = log_spectrum
        self.target_shape = target_shape
        self.scale_factor = scale_factor
        self.eps = eps
        super(SpectrumBuilder, self).__init__(
            **kwargs
        )

    def preprocessing(self, image, kspace):
        image = image * self.scale_factor
        spectrum = ortho_fft2d(image)
        spectrum = adjust_image_size(spectrum, self.target_shape)[..., None]
        if self.log_spectrum:
            log_spectrum = tf.math.log(tf.abs(spectrum) + self.eps)
            results = (spectrum[0], log_spectrum[0])
        else:
            results = spectrum[0]
        if self.return_images:
            return results, adjust_image_size(image, self.target_shape)[0, ..., None]
        else:
            return results


class JustCropScaleData(FastMRIDatasetBuilder):
    def __init__(self, crop_image_data=True, image_size=(320, 320), scale_factor=1e6, **kwargs):
        self.image_size = image_size
        self.crop_image_data = crop_image_data
        self.scale_factor = scale_factor
        super(JustCropScaleData, self).__init__(**kwargs)

    def preprocessing(self, image, kspace):
        image = image * self.scale_factor
        kspace = kspace * self.scale_factor
        kspace = kspace[0]
        if self.crop_image_data:
            return kspace, adjust_image_size(image, self.image_size)[..., None]
        else:
            output_shape = tf.shape(image)[1:][None, :]
            output_shape = tf.tile(output_shape, [tf.shape(image)[0], 1])
            return (kspace, output_shape), image[..., None]