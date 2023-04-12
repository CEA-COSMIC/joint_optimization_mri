import tensorflow as tf
from fastmri_recon.models.subclassed_models.vnet import VnetComplex


class StackVNet(tf.keras.Model):
    """V-Net with stack training for memory utilization
    """
    def __init__(self, n_filters=16, n_scales=3, vnet_kwargs={}, **kwargs):
        self.vnet_kwargs = dict(vnet_kwargs)
        self.vnet_kwargs['layers_n_channels'] = [n_filters * 2 ** i for i in range(n_scales)]
        super(StackVNet, self).__init__(**kwargs)
        self.vnet = VnetComplex(
            **self.vnet_kwargs
        )
    
    def call(self, input):
        return self.vnet(input)[..., 0]
        