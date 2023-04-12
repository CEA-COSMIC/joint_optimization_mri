import tensorflow as tf


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_decode():
    return tf.io.FixedLenFeature(shape=(), dtype=tf.string)


def serialize_tensor(tensor):
    if isinstance(tensor, tuple):
        return tuple(serialize_tensor(t) for t in tensor)
    else:
        return bytes_feature(tf.io.serialize_tensor(tensor).numpy())
