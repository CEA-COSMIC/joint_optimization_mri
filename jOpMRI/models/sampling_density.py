import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.models.subclassed_models.xpdnet import XPDNet

from .acquisition.loupe import LOUPE


class JointModel(Model):
    def __init__(self, image_size, multicoil=False, acq_kwargs={}, recon_kwargs={}, **kwargs):
        super(JointModel, self).__init__(**kwargs)
        self.multicoil = multicoil
        self.image_size = image_size
        if acq_kwargs['type'] == 'loupe':
            self.acq_model = LOUPE(**acq_kwargs['params'])
        else:
            raise ValueError('Bad acquisitin model')
        if recon_kwargs['type'] == 'xpdnet':
            recon_params = recon_kwargs['params']
            model_fun, kwargs, n_scales, res = [
            (model_fun, kwargs, n_scales, res)
            for m_name, m_size, model_fun, kwargs, _, n_scales, res in get_model_specs(n_primal=recon_params['n_primal'], force_res=False)
            if m_name == recon_params['name'] and m_size == recon_params['size']
            ][0]
            run_params = {
                'n_primal': recon_params['n_primal'],
                'multicoil': self.multicoil,
                'n_scales': n_scales,
                'n_iter': recon_params['n_iter'],
            }
            self.recon_net = XPDNet(model_fun, kwargs, **run_params)
            if recon_kwargs['weights_file'] is not None:
                self.init_recon(recon_kwargs['weights_file'])
        else:
            raise ValueError('Bad reconstruction model')

    def init_recon(self, weights):
        self.call(tf.zeros((self.batch_size, *self.image_size)))
        self.recon_net.load_weights(weights)
        # By default, we dont learn in case we load a network.
        self.recon_net.trainable = False

    def call(self, input):
        mask = self.acq_model(input)
        recon = self.recon_net([input, mask])
        return recon