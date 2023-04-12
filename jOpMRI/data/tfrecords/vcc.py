import glob, os
from functools import partial

import tensorflow as tf

from .utils import serialize_tensor, feature_decode

TENSOR_DTYPES = {
    'images': tf.complex64,
    'true_abs_image': tf.float32,
}


def encode_vcc_example(model_inputs):
    model_inputs = [serialize_tensor(mi) for mi in model_inputs]
    feature = {
        'images': model_inputs[0],
        'true_abs_image': model_inputs[1],
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def set_shapes_vcc(data_dict, is2D=True):
    for k, v in data_dict.items():
        if k == 'images':
            if is2D:
                v.set_shape([None, None, None])
            else:
                v.set_shape([1, None, None, None])
        elif k == 'true_abs_image':
            if is2D:
                v.set_shape([None, None, None, 1])
            else:
                v.set_shape([1, None, None, None])
    return data_dict


def decode_vcc_example(raw_record, slice_random=True, is2D=True, normalize=False, im_size=None):
    features = {
        'images': feature_decode(),
        'true_abs_image': feature_decode(),
    }
    example = tf.io.parse_example(raw_record, features=features)
    example_parsed = {
        k: tf.io.parse_tensor(tensor, TENSOR_DTYPES[k])
        for k, tensor in example.items()
    }
    example_parsed = set_shapes_vcc(example_parsed, is2D=is2D)
    if slice_random:
        num_slices = tf.shape(example_parsed['images'])[0]
        slice_start = tf.random.uniform([1], maxval=num_slices, dtype=tf.int32)[0]
        model_inputs = example_parsed['images'][slice_start:slice_start+1]
        model_outputs = example_parsed['true_abs_image'][slice_start:slice_start+1]
    else:
        model_inputs = example_parsed['images']
        model_outputs = example_parsed['true_abs_image']
    if normalize:
        model_inputs = model_inputs / tf.cast(tf.math.reduce_mean(tf.math.abs(model_inputs)), model_inputs.dtype)
        model_outputs = model_outputs / tf.cast(tf.math.reduce_mean(tf.math.abs(model_outputs)), model_outputs.dtype)
    if im_size is not None:
        @tf.function
        def _pad(im, target_shape):
            cur_shape = tf.shape(im)
            diff = target_shape - cur_shape
            pad_sizes = diff // 2
            return tf.pad(im, tf.repeat(pad_sizes[:, None], (2,), axis=-1))
        model_inputs = _pad(model_inputs, im_size)
        model_outputs = _pad(model_outputs, im_size)
    if is2D:
        # FIXME: This wont work for 3D
        model_inputs = tf.transpose(model_inputs, perm=[0, 2, 1])
        model_outputs = tf.transpose(model_outputs, perm=[0, 2, 1, 3])
    return model_inputs, model_outputs


def filter_contrast(filename, contrast):
    if contrast is None:
        return True
    elif isinstance(contrast, str):
        return contrast in filename
    else:
        for c in contrast:
            if c in filename:
                return True
        return False


def read_data(path, contrast=None, split_slices=True, n_samples=None, slice_random=True, use_abs_image_input=False,
              send_filenames=False, rotate_dataset=True, is2D=True, create_y_as_abs=True, normalize=False, cardinality=None,
              im_size=None, scale_factor=1):
    files = glob.glob(os.path.join(path, '*.tfr')) # 'file_brain_AXT1_201_6002688.tfr'))
    filtered_files = [file for file in files if filter_contrast(file, contrast)]
    filenames = tf.data.Dataset.from_tensor_slices(filtered_files)
    dataset = tf.data.TFRecordDataset(
        filtered_files,
        num_parallel_reads=1,
    ).apply(tf.data.experimental.ignore_errors(log_warning=True))
    dataset = dataset.map(
        partial(decode_vcc_example, slice_random=slice_random, is2D=is2D, normalize=normalize, im_size=[1, *im_size] if im_size is not None else None),
    )
    if scale_factor != 1:
        dataset = dataset.map(lambda x, y: (x*scale_factor, y*scale_factor))
    if send_filenames:
        dataset = dataset.zip((dataset, filenames))
        dataset = dataset.map(
            lambda x, filename: (x, tf.repeat(filename, tf.shape(x[0])[0])),
        )
    if split_slices:
        dataset = dataset.unbatch()
        # currently we need to batch to 1 for consistency
        dataset = dataset.batch(1)
    if cardinality is not None:
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
    if n_samples is not None:
        dataset = dataset.take(n_samples)
    if not send_filenames and create_y_as_abs:
        dataset = dataset.map(
            lambda complex_image, _: (complex_image, tf.abs(complex_image)[..., None]),
        )
        if use_abs_image_input:
            dataset = dataset.map(
                lambda complex_image, _: (tf.abs(complex_image), tf.abs(complex_image)[..., None]),
            )
    return dataset
