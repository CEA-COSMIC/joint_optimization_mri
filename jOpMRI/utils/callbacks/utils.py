import logging
import os, gc
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from .tensorboard import TrajectoryViewer, GradientViewer, CustomTBCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import MultiOptimizer
import pickle as pkl
import logging

log = logging.getLogger(__name__)

class ClearMemory(tf.keras.callbacks.Callback):
    def __init__(self, update_freq=1, **kwargs):
        super(ClearMemory, self).__init__(**kwargs)
        self.update_freq = update_freq
        
    def on_epoch_end(self, epoch, logs=None):
        if not(epoch % self.update_freq):
            log.info('Clearing memory')
            gc.collect()
            tf.keras.backend.clear_session()

class ModelCheckpointWorkAround(ModelCheckpoint):
    def __init__(self, filepath, save_optimizer=True, **kwargs):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.save_optimizer = save_optimizer
        super(ModelCheckpointWorkAround, self).__init__(filepath=filepath, **kwargs)

    def set_model(self, model):
        # Work around, so that the if at
        # https://github.com/tensorflow/tensorflow/blob/1186e3f2098793952aa82bf356dfe51b967fb26c/tensorflow/python/keras/callbacks.py#L1189
        # is skipped, so that self.save_weights_only remains False.
        self.model = model

    def _save_model(self, epoch, batch, logs):
        # Save the model with super
        super(ModelCheckpointWorkAround, self)._save_model(epoch=epoch, batch=batch, logs=logs)
        if self.save_optimizer:
            # Save the optimizer
            folder = os.path.dirname(self._get_file_path(epoch, batch, logs))
            if isinstance(self.model.optimizer, MultiOptimizer):
                weights = [opt['optimizer'].get_weights() for opt in self.model.optimizer.optimizer_specs]
            else:
                weights = self.model.optimizer.get_weights()
            with open(os.path.join(folder, 'optimizer_E{epoch:02d}.pkl'.format(epoch=epoch + 1)), 'wb') as f:
                pkl.dump(weights, f)


def get_callbacks(outdir, save_freq, image_freq=None, grad_freq=None, clean_freq=None, profile_batch=0, get_gmax_smax_metrics=False):
    log_dir = os.path.join(outdir, 'tensorboard')
    os.makedirs(log_dir, exist_ok=True)
    tboard_cback = CustomTBCallback(
        profile_batch=profile_batch,
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        get_penalty_metrics=get_gmax_smax_metrics,
    )
    chkpt_path = os.path.join(outdir, 'checkpoints')
    Path(chkpt_path).mkdir(parents=True, exist_ok=True)
    chkpt_cback = ModelCheckpointWorkAround(
        os.path.join(chkpt_path, 'model_E{epoch:02d}_L{loss:1.3f}.h5'), # _SSIM{keras_ssim:1.3f} Add back if needed
        save_freq=save_freq,
        save_weights_only=True,
    )
    extra_outputs = ()
    if image_freq is not None:
        extra_outputs += (TrajectoryViewer(log_dir=log_dir, update_freq=image_freq),)
    if grad_freq is not None:
        extra_outputs += (GradientViewer(log_dir=log_dir, update_freq=grad_freq),)
    if clean_freq is not None:
        extra_outputs += (ClearMemory(update_freq=clean_freq),)
    return [chkpt_cback, tboard_cback, *extra_outputs]