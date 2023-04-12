from tfkbnufft.mri.dcomp_calc import calculate_density_compensator
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tensorflow.keras.regularizers import Regularizer
import tensorflow as tf
import tensorflow_nufft as tfft
from sparkling.utils.trajectory import get_time_vector
from jOpMRI.trajectory.utils import get_grads_n_slew, interpolate_shots_only
import numpy as np


def redo_till_no_nan(func, max_iter=10):
    """
    Redo a function until it doesn't return NaNs.
    """
    def func_wrapper(*args, **kwargs):
        for i in tf.range(max_iter):
            if i >= 1:
                tf.print("Found NaN! Repeating ")
                tf.print(i)
                tf.print(func)
                tf.print(tf.math.reduce_any(tf.math.is_nan(tf.abs(args[0]))))
                #np.save("nan_output.npy", (args, kwargs))
            out = func(*args, **kwargs)
            if not tf.math.reduce_any(tf.math.is_nan(tf.abs(out))):
                break
        return out
    return func_wrapper 

@redo_till_no_nan 
def nufft(source, points, grid_shape=None, transform_type='type_2', fft_direction='forward', tol=1e-3):
    if grid_shape is None:
        im_size = tf.shape(source)[-points.shape[-1]:]
    else:
        im_size = grid_shape
    options = tfft.Options()
    if points.shape[-1] == 3:
        options.max_batch_size = 1
    return tfft.nufft(
        source=source,
        points=points,
        grid_shape=grid_shape,
        transform_type=transform_type,
        fft_direction=fft_direction, 
        tol=tol,
        options=options,
    ) / tf.math.sqrt(tf.math.reduce_prod(tf.cast(im_size, tf.complex64) * 2))

def _next_smooth_int(n):
  """Find the next even integer with prime factors no larger than 5.
  Args:
    n: An `int`.
  Returns:
    The smallest `int` that is larger than or equal to `n`, even and with no
    prime factors larger than 5.
  """
  if n <= 2:
    return 2
  if n % 2 == 1:
    n += 1    # Even.
  n -= 2      # Cancel out +2 at the beginning of the loop.
  ndiv = 2    # Dummy value that is >1.
  while ndiv > 1:
    n += 2
    ndiv = n
    while ndiv % 2 == 0:
      ndiv /= 2
    while ndiv % 3 == 0:
      ndiv /= 3
    while ndiv % 5 == 0:
      ndiv /= 5
  return n


@tf.function
def get_density_compensators(trajectory, im_size=(320, 320), implementation='tensorflow_nufft', interpob=None):
    @tf.custom_gradient
    def _zero_grad_density_compensators(traj):
        batch_shape = tf.shape(traj)[:-2]
        grid_shape = tf.TensorShape([_next_smooth_int(i*2) for i in im_size]) # Canonicalize.
        weights = tf.ones(
            tf.concat([batch_shape, tf.shape(traj)[-2:-1]], 0),
            dtype=tf.float32
        )
        def run_loop(weights):
            for i in tf.range(10):
                weights /= tf.abs(tfft.interp(
                    tfft.spread(
                        tf.cast(weights, tf.complex64),
                        traj,
                        grid_shape,
                        tol=1e-3,
                    ), 
                    traj,
                    tol=1e-3
                ))
            return weights
        weights = run_loop(weights) 
        test_im = tf.ones(im_size, dtype=tf.complex64)
        test_im_recon = nufft(
            tf.cast(weights, tf.complex64) * nufft(
                test_im,
                traj,
                transform_type='type_2',
                fft_direction='forward',
                tol=1e-3,
            ),
            traj,
            grid_shape=im_size,
            transform_type='type_1',
            fft_direction='backward',
            tol=1e-3,
        )
        ratio = tf.reduce_mean(tf.math.abs(test_im_recon))
        weights = weights / tf.cast(ratio, weights.dtype)
        def grad(dtraj):
            return None
        return weights, grad
    if implementation == 'tensorflow-nufft':
        return _zero_grad_density_compensators(tf.transpose(trajectory))
    elif implementation == 'tfkbnufft':
        return calculate_density_compensator(
            interpob,
            kbnufft_forward(interpob),
            kbnufft_adjoint(interpob),
            trajectory,
            zero_grad=True,
        )


class SampleRepulsion(Regularizer):
    def __init__(self, kTE=1, temporal_repulsion=True, **kwargs):
        self.kTE = kTE
        self.temporal_repulsion = temporal_repulsion
        super(SampleRepulsion, self).__init__(**kwargs)

    def __call__(self, x, **kwargs):
        [Nc, Ns, D] = x.shape
        x = tf.reshape(x, (-1, 2))
        distance = tf.sqrt(tf.math.reduce_sum((x[None, :] - x[:, None])**2, axis=-1) + 1e-5)
        if self.temporal_repulsion:
            init_time = tf.cast(get_time_vector(Ns, self.kTE), tf.float32)
            max_time = tf.math.reduce_max(init_time)-tf.math.reduce_min(init_time)
            time = tf.tile(init_time, [Nc])
            # Below temporal weighting comes from SPARKLING
            time_diff = tf.exp(tf.abs(time[None, :] - time[:, None]) / max_time * 10**D / Nc / Ns / 2)
            distance = distance * time_diff
        neg_loss = 1/2 * tf.math.reduce_mean(distance)
        # Sample repulsion is - ||x||_2 norm. More distance --> lesser loss
        return -neg_loss
    
class ConstraintPenalty(Regularizer):
    def __init__(self, scale, img_size, FOV, gyromagnetic_constant, gradient_raster_time, GMax, SMax, num_samples_per_shot, **kwargs):
        self.img_size = img_size
        self.FOV = FOV
        self.GMax = GMax
        self.SMax = SMax
        self.scale = scale
        self.gyromagnetic_constant = gyromagnetic_constant
        self.gradient_raster_time = gradient_raster_time
        self.num_samples_per_shot = num_samples_per_shot
        super(ConstraintPenalty, self).__init__(**kwargs)

    def get_all_loss(self, x, **kwargs):
        traj = interpolate_shots_only(x, target_Ns=self.num_samples_per_shot) * 2 * np.pi
        G, S, K0 = get_grads_n_slew(
            traj,
            img_size=self.img_size,
            FOV=self.FOV,
            gyromagnetic_constant=self.gyromagnetic_constant,
            gradient_raster_time=self.gradient_raster_time,
        )
        grad_loss = self.scale * tf.math.reduce_sum(tf.cast(G>self.GMax, tf.float32)*(G-self.GMax))
        slew_loss = self.scale * tf.math.reduce_sum(tf.cast(S>self.SMax, tf.float32)*(S-self.SMax))
        te_loss = self.scale * tf.math.reduce_sum(tf.abs(K0))
        max_grad = tf.math.reduce_max(G)
        max_slew = tf.math.reduce_max(S)
        return (grad_loss, slew_loss, te_loss), (max_grad, max_slew)

    def __call__(self, x, **kwargs):
        (grad_loss, slew_loss, te_loss), (max_grad, max_slew) = self.get_all_loss(x, **kwargs)
        return grad_loss + slew_loss + te_loss