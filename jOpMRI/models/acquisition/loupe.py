"""
    Layers for LOUPE
    
    For more details, please read:
    
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019. https://arxiv.org/abs/1901.01960.
"""


# third party
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf


class RescaleProbMap(Layer):
    """
    Rescale Probability Map

    given a prob map x, rescales it so that it obtains the desired sparsity

    if mean(x) > sparsity, then rescaling is easy: x' = x * sparsity / mean(x)
    if mean(x) < sparsity, one can basically do the same thing by rescaling 
                            (1-x) appropriately, then taking 1 minus the result.
    """

    def __init__(self, sparsity, **kwargs):
        # TODO move this to be controlled by a tf.Variable so that it is
        # stored when model is saved
        self.sparsity = sparsity
        super(RescaleProbMap, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RescaleProbMap, self).build(input_shape)

    def call(self, x):
        xbar = K.mean(x)
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)

        # compute adjucement
        le = tf.cast(tf.less_equal(r, 1), tf.float32)
        return le * x * r + (1-le) * (1 - (1 - x) * beta)


class ProbMask(Layer):
    """ 
    Probability mask layer
    Contains a layer of weights, that is then passed through a sigmoid.

    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """

    def __init__(self, slope=10,
                 initializer=None,
                 slope_trainable=False,
                 **kwargs):
        """
        note that in v1 the initial initializer was uniform in [-A, +A] where A is some scalar.
        e.g. was RandomUniform(minval=-2.0, maxval=2.0, seed=None),
        But this is uniform *in the logit space* (since we take sigmoid of this), so probabilities
        were concentrated a lot in the edges, which led to very slow convergence, I think.

        IN v2, the default initializer is a logit of the uniform [0, 1] distribution,
        which fixes this issue
        """
        self.slope_value = slope
        if initializer == None:
            self.initializer = self._logit_slope_random_uniform
        else:
            self.initializer = initializer

        # higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it
        #   during training if necessary
        self.slope = tf.Variable(
            slope, dtype=tf.float32, trainable=slope_trainable)
        super(ProbMask, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        takes as input the input data, which is [N x ... x 2] 
        """
        # Create a trainable weight variable for this layer.
        lst = list(input_shape)
        input_shape_h = tuple(lst)

        self.mult = self.add_weight(
            name='logit_weights',
            shape=input_shape_h[1:],
            initializer=self.initializer,
            trainable=True
        )

        # Be sure to call this somewhere!
        super(ProbMask, self).build(input_shape)

    def call(self, x):
        logit_weights = 0*x[..., 0:1] + self.mult
        return tf.sigmoid(self.slope * logit_weights)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _logit_slope_random_uniform(self, shape, dtype=None, eps=0.01):
        # eps could be very small, or somethinkg like eps = 1e-6
        #   the idea is how far from the tails to have your initialization.
        x = K.random_uniform(shape, dtype=dtype, minval=eps,
                             maxval=1.0-eps)  # [0, 1]

        # logit with slope factor
        return - tf.math.log(1. / x - 1.) / self.slope


class ThresholdRandomMask(Layer):
    """ 
    Local thresholding layer

    Takes as input the input to be thresholded, and the threshold

    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """

    def __init__(self, slope=12, slope_trainable=False, **kwargs):
        """
        if slope is None, it will be a hard threshold.
        """
        # higher slope means a more step-function-like logistic function
        # note: slope is converted to a tensor so that we can update it
        #   during training if necessary
        self.slope = None
        if slope is not None:
            self.slope = tf.Variable(
                slope, dtype=tf.float32, trainable=slope_trainable)
        super(ThresholdRandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ThresholdRandomMask, self).build(input_shape)

    def call(self, x):
        inputs = x[0]
        thresh = x[1]
        if self.slope is not None:
            return tf.sigmoid(self.slope * (inputs-thresh))
        else:
            return inputs > thresh


class RandomMask(Layer):
    """ 
    Create a random binary mask of the same size as the input shape
    """

    def __init__(self, **kwargs):
        super(RandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RandomMask, self).build(input_shape)

    def call(self, x):
        input_shape = tf.shape(x)
        threshs = K.random_uniform(
            input_shape, minval=0.0, maxval=1.0, dtype='float32')
        return threshs


class LOUPE(Model):
    def __init__(self, pmask_slope, pmask_init, sparsity, sample_slope, **kwargs):
        super(LOUPE, self).__init__(**kwargs)
        self.prob_mask = ProbMask(name='prob_mask', slope=pmask_slope, initializer=pmask_init)
        self.prob_density = RescaleProbMap(sparsity, name='prob_density')
        self.threshold = RandomMask(name='random_mask_thresholder')
        self.masker = ThresholdRandomMask(slope=sample_slope, name='sampled_mask')

    def call(self, input):
        prob_mask_tensor = self.prob_mask(tf.cast(input, 'float32'))
        prob_mask_tensor = self.prob_density(prob_mask_tensor)
        thresh_tensor = self.threshold(prob_mask_tensor)
        mask = self.masker([prob_mask_tensor, thresh_tensor])[..., 0]
        return mask