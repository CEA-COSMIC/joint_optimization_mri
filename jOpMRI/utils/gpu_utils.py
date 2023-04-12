import tensorflow as tf


def configure_all_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)
