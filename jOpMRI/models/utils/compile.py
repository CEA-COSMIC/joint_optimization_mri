import tensorflow_probability as tfp
import warnings
from functools import partial
import os
import tensorflow as tf
import tensorflow_addons as tfa
import pickle as pkl
import logging
from .schdulers import LinearScheduler

log = logging.getLogger(__name__)


def normalize_data(y_true, y_pred):
    shape_y_pred = tf.shape(y_pred)
    y_pred = tf.reshape(y_pred, [shape_y_pred[0], -1])
    y_true = tf.reshape(y_true, [shape_y_pred[0], -1])
    y_true_mean = tf.reduce_mean(y_true, axis=1)
    y_pred_mean = tf.reduce_mean(y_pred, axis=1)
    factor = y_true_mean / y_pred_mean
    y_pred *= factor[:, None]
    return tf.reshape(y_pred, shape_y_pred)


def keras_psnr(y_true, y_pred, norm_data=True):
    y_true = tf.abs(y_true)
    y_pred = tf.abs(y_pred)
    if norm_data:
        y_pred = normalize_data(y_true, y_pred)
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel - min_pixel)


def keras_ssim(y_true, y_pred, norm_data=True):
    y_true = tf.abs(y_true)
    y_pred = tf.abs(y_pred)
    if norm_data:
        y_pred = normalize_data(y_true, y_pred)
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.ssim(y_true, y_pred, max_pixel - min_pixel)


def normalized_mse(y_true, y_pred):
    y_pred = normalize_data(y_true, y_pred)
    loss = tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred))
    return loss


def default_model_compile(model, lr, loss='compound_mssim'):
    if loss == 'compound_mssim':
        loss = compound_l1_mssim_loss
    elif loss == 'mssim':
        loss = partial(compound_l1_mssim_loss, alpha=0.9999)
        loss.__name__ = "mssim"
    elif loss == 'normalized_mse':
        loss = normalized_mse
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=loss,
        metrics=[keras_psnr, keras_ssim],
    )


def default_multiopt_compile(jOpModel, loss='compound_mssim', reuse_traj_opt=False, reuse_recon_opt=False, 
                             total_steps=None, total_decim=None, **kwargs):
    if loss == 'compound_mssim':
        loss = compound_l1_mssim_loss
    elif loss == 'mssim':
        loss = partial(compound_l1_mssim_loss, alpha=0.9999)
        loss.__name__ = "mssim"
    elif loss == 'normalized_mse':
        loss = normalized_mse
    elif loss == 'compound_l1_l2':
        loss = compound_l1_l2
    elif loss == 'zero':
        loss = lambda y_true, y_pred: 0
    elif loss == 'l1':
        loss = tf.keras.losses.MeanAbsoluteError()
    if reuse_traj_opt:
        adam_opt = jOpModel.optimizer.optimizer_specs[0]['optimizer']
    else:
        if 'traj_opt_kwargs' not in kwargs or 'recon_opt_kwargs' not in kwargs:
            # Backward compatibility
            kwargs['traj_opt_kwargs'] = {
                'learning_rate': kwargs['lr'],
                'scheduler': None,
            }
            kwargs['recon_opt_kwargs'] = {
                'learning_rate': kwargs['lr'],
            }
        kwargs['traj_opt_kwargs'] = dict(kwargs['traj_opt_kwargs'])
        if kwargs['traj_opt_kwargs']['scheduler'] == 'linear':
            kwargs['traj_opt_kwargs']['learning_rate'] = LinearScheduler(
                initial_learning_rate=kwargs['traj_opt_kwargs']['learning_rate'],
                final_learning_rate=kwargs['traj_opt_kwargs']['learning_rate']/total_decim,
                decay_steps=total_steps,
            )
        elif kwargs['traj_opt_kwargs']['scheduler'] == 'exponential':
            if total_decim is None:
                total_decim = 1/jOpModel.optimizer.optimizer_specs[0]['optimizer'].lr.decay_rate
                total_steps = jOpModel.optimizer.optimizer_specs[0]['optimizer'].lr.decay_steps
            kwargs['traj_opt_kwargs']['learning_rate'] = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=kwargs['traj_opt_kwargs']['learning_rate'],
                decay_steps=total_steps,
                decay_rate=1/total_decim,
            )
        del kwargs['traj_opt_kwargs']['scheduler']
        adam_opt = tf.keras.optimizers.Adam(**kwargs['traj_opt_kwargs'])
    if reuse_recon_opt:
        rdam_opt = jOpModel.optimizer.optimizer_specs[1]['optimizer']
    else:
        rdam_opt = tfa.optimizers.RectifiedAdam(**kwargs['recon_opt_kwargs'])
    optimizers_n_layers = [
        (adam_opt, jOpModel.base_traj),
        (rdam_opt, jOpModel.recon_net),
    ]
    # We only add metrics of SSIM and PSNR only when needed
    jOpModel.compile(
        optimizer=tfa.optimizers.MultiOptimizer(optimizers_n_layers),
        loss=loss,
        metrics=[] if jOpModel.only_traj_related_learn else [keras_psnr, keras_ssim, normalized_mse],
    )


def upscale_traj_optimizer(jOpModel, old_weights=None, interpolate_type='upscale'):
    traj_opt = jOpModel.optimizer.optimizer_specs[0]['optimizer']
    weights_given = True
    if old_weights is None:
        old_weights = traj_opt.get_weights()
        weights_given = False
    # zero grad to get new weights
    traj_opt.apply_gradients(zip([tf.zeros_like(w) for w in jOpModel.trainable_variables], jOpModel.trainable_variables))
    new_weights = traj_opt.get_weights()
    for i, opt_w in enumerate(new_weights):
        if opt_w.shape == jOpModel.base_traj.trajectory.shape:
            log.info("Upscaling optimizer weight")
            # FIXME : remove hardcode, works only with ADAM
            if interpolate_type in ['upscale', 'zero_fill']:
                new_weights[i] = tfp.math.interp_regular_1d_grid(
                    x=tf.cast(tf.linspace(0, 1, jOpModel.base_traj.trajectory.shape[1]), opt_w.dtype),
                    x_ref_min=0,
                    x_ref_max=1,
                    y_ref=old_weights[i] if weights_given else old_weights[i-int((len(new_weights)-1)/2)],
                    axis=1
                )
                if interpolate_type == 'zero-fill':
                    new_weights[i][:, ::2] = 0
    jOpModel.optimizer.optimizer_specs[0]['optimizer'].set_weights(new_weights)
    

def compound_l1_mssim_loss(y_true, y_pred, alpha=0.995):
    y_pred = normalize_data(y_true, y_pred)
    mssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=tf.reduce_max(y_true))
    if len(tf.shape(mssim))==2:
        mssim = mssim[:, 0]
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
    l2 = tf.reduce_mean(tf.abs(y_true - y_pred)**2/2)
    loss = alpha * (1 - mssim) + (1 - alpha) * l1 + (1 - alpha)**2 * l2
    return loss


def compound_l1_l2(y_true, y_pred):
    y_pred = normalize_data(y_true, y_pred)
    l1 = tf.reduce_mean(tf.reshape(tf.abs(y_true - y_pred), [tf.shape(y_true)[0], -1]), axis=1)
    l2 = tf.reduce_mean(tf.reshape(tf.abs(y_true - y_pred)**2, [tf.shape(y_true)[0], -1]), axis=1)
    loss = l1 + l2
    tf.print(loss)
    return loss 


def load_optimizer(model, opt_kwargs, optimizer_file, load_acq=True, load_recon=True):
    default_multiopt_compile(model, **opt_kwargs)
    old_trainable = model.base_traj.trainable, model.recon_net.trainable
    model.base_traj.trainable, model.recon_net.trainable = load_acq, load_recon
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
    if os.path.exists(optimizer_file):
        weights = pkl.load(open(optimizer_file, 'rb'))
        if weights != []:
            try:
                if load_acq:
                    model.optimizer.optimizer_specs[0]['optimizer'].set_weights(weights[0])
            except:
                if old_trainable[0]:
                    raise ValueError("Could not load acquisition optimizer weights!")
                else:
                    log.warning("Could not load acquisition optimizer weights!, but I think its ok as we are not training it")
            try:    
                if load_recon:
                    model.optimizer.optimizer_specs[1]['optimizer'].set_weights(weights[1])
            except:
                if old_trainable[1]:
                    raise ValueError("Could not load reconstruction optimizer weights!")
                else:
                    log.warning("Could not load reconstruction optimizer weights!, but I think its ok as we are not training it")
    else:
        warnings.warn('Didnt find optimizer to load')
    model.base_traj.trainable, model.recon_net.trainable = old_trainable