import tensorflow as tf
import numpy as np


class LinearScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, final_learning_rate, decay_steps):
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.decay_steps = decay_steps

    def __call__(self, step):
        lr = tf.cast(step, tf.float32) / tf.cast(self.decay_steps, tf.float32) * (self.final_learning_rate - self.initial_learning_rate) + self.initial_learning_rate
        return lr

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'final_learning_rate': self.final_learning_rate,
            'decay_steps': self.decay_steps,
        }
        
def get_total_epochs(cfg):
    if cfg['train']['type'] == 'multi_resolution':
        total_epochs = cfg['train']['num_epochs']*(int(np.log2(cfg['trajectory']['algo_params']['start_decim'])) + 1)
    elif cfg['train']['type'] == 'segmented':
        total_epochs = 2 + 2 * (cfg['train']['num_epochs']*(int(np.log2(cfg['trajectory']['algo_params']['start_decim'])) + 1))
    else:
        total_epochs = cfg['train']['num_epochs']
    return total_epochs