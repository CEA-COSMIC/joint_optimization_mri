import io
from locale import normalize
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow_addons.optimizers import MultiOptimizer
import matplotlib.pyplot as plt

from sparkling.utils.plotting import scatter_shots
from jOpMRI.utils.trajectory import view_gradients


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


class CustomTBCallback(TensorBoard):
    def __init__(self, get_penalty_metrics=False, *args, **kwargs):
        self.get_penalty_metrics = get_penalty_metrics
        super().__init__(*args, **kwargs)
    def _log_epoch_metrics(self, epochs, logs):
        if self.get_penalty_metrics:
            penalty_losses, (cur_gmax, cur_smax) = self.model.base_traj.get_penalty_losses()
            for penalty_loss in zip(penalty_losses, ['grad', 'slew', 'te']):
                logs['penalty_' + penalty_loss[1]] = penalty_loss[0].numpy()
            logs['gmax'] = cur_gmax.numpy()
            logs['smax'] = cur_smax.numpy()
            if self.model.cfg['trajectory']['algo_params']['trajectory_regularization'] == 'constraint_penalty':
                # Also show just the recon loss for simplicity
                logs['recon_loss'] = logs['loss'] - tf.math.reduce_sum(penalty_losses).numpy()
        for layer in self.model.layers:
            if layer.losses:
                logs["reg_%s" % layer.name] = tf.reduce_sum(layer.losses)
        super()._log_epoch_metrics(epochs, logs)


class TrajectoryViewer(Callback):
    def __init__(self, log_dir, update_freq=1, **kwargs):
        self.file_writer_cm = tf.summary.create_file_writer(log_dir + '/traj')
        self.update_freq = update_freq
        super(TrajectoryViewer, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if not(epoch % self.update_freq):
            trajectory = self.model.base_traj.trajectory.numpy()
            dimension = trajectory.shape[-1]
            figure = scatter_shots(
                trajectory,
                num_shots='all' if dimension == 2 else 200,
                return_fig=True,
                fig_scale=2,
                plot_samples=False,
                temporal_color=False,
                random_seed=0,
                random=True,
            )
            cm_image = plot_to_image(figure)
            with self.file_writer_cm.as_default():
                tf.summary.image("Trajectory", cm_image, step=epoch)

class GradientViewer(Callback):
    def __init__(self, log_dir, update_freq=1, log=False, **kwargs):
        self.file_writer_cm = tf.summary.create_file_writer(log_dir + '/grad_traj')
        self.update_freq = update_freq
        self.log = log
        super(GradientViewer, self).__init__(**kwargs)

    def on_batch_end(self, batch, logs=None):
        if not(batch % self.update_freq):
            trajectory = self.model.base_traj.trajectory
            try:
                if isinstance(self.model.optimizer, MultiOptimizer):
                    grads = self.model.optimizer.optimizer_specs[0]['gv'][0][0]
                else:
                    grad = self.model.optimizer['gv']
                figure = view_gradients(
                    trajectory,
                    grads=grads,
                    fig_scale=1.2,
                    return_figure=True,
                    log=self.log,
                    num_shots='all' if trajectory.shape[-1] == 2 else 200,
                    random_seed=0,
                    length=0.05,
                    normalize=True,
                )
                cm_image = plot_to_image(figure)
                with self.file_writer_cm.as_default():
                    tf.summary.image("GradientTrajectory", cm_image, step=batch)
            except:
                print("Failed to handle for " + str(self.model.optimizer.optimizer_specs[0]['gv'][0][0]))
                pass
