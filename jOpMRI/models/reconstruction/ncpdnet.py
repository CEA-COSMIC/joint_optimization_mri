import tensorflow as tf
from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet

from jOpMRI.models.preprocess.recon_preprocess import PreProcModel


class ReconModel(tf.keras.Model):
    def __init__(self, preproc_params, recon_params, image_size, **kwargs):
        super(ReconModel, self).__init__(**kwargs)
        self.preproc = PreProcModel(**preproc_params)
        self.multicoil = preproc_params['multicoil']
        self.preproc.trainable = False
        self.recon_net = NCPDNet(
            multicoil=self.multicoil,
            dcomp=recon_params['dcomp'],
            im_size=image_size,
            **recon_params['params'],
        )

    def call(self, inputs):
        model_inputs = self.preproc(inputs)
        recon_out = self.recon_net(model_inputs)
        recon_out = recon_out[:, None, ...]
        return recon_out
