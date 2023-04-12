import warnings
import os
import numpy as np
import tensorflow as tf
import pydicom as pdcm
from tf_fastmri_data.preprocessing_utils.fourier.cartesian import ortho_ifft2d, ortho_fft2d
from mapvbvd import mapVBVD


def load_magnitude_from_dicom(folder_path):
    folder_path = folder_path.numpy().decode('utf-8')
    fm_dcms = [
        os.path.join(folder_path, filename)
        for filename in sorted(os.listdir(folder_path))
    ]
    # Transform the stored data into field map
    mag_images = np.array(
        [pdcm.dcmread(fm_dcm).pixel_array for fm_dcm in fm_dcms])
    mag_images = [np.rot90(f.T, k=-1) for f in mag_images][::-1]
    mag_images = np.stack(mag_images, axis=-1)
    return mag_images


def load_phase_from_dicom(folder_path):
    folder_path = folder_path.numpy().decode('utf-8')
    fm_dcms = [
        os.path.join(folder_path, filename)
        for filename in sorted(os.listdir(folder_path))
    ]
    # Get the scale
    fm_data = pdcm.dcmread(fm_dcms[0])
    slope, intercept = fm_data.RescaleSlope, fm_data.RescaleIntercept
    fm_range = 2 ** fm_data.BitsStored
    # Transform the stored data into phase_map
    phase_map = np.array(
        [pdcm.dcmread(fm_dcm).pixel_array for fm_dcm in fm_dcms])
    phase_map = [np.rot90(f.T, k=-1) for f in phase_map][::-1]
    phase_map = np.stack(phase_map, axis=-1)
    phase_map = (slope * phase_map + intercept) / slope / fm_range
    phase_map = phase_map * 2 * np.pi
    return phase_map


def load_twix(filename, is2D=True):
    # TODO fix it for 3D, it is hardcoded to 2D
    twixObj = mapVBVD(filename.numpy())
    if isinstance(twixObj, list):
        twixObj = twixObj[-1]
    twixObj.image.flagRemoveOS = False
    twixObj.image.squeeze = True
    raw_kspace = twixObj.image['']
    [num_adc_samples, channels, num_shots, num_slices] = tf.shape(raw_kspace)
    if is2D:
        rel_slice_order = [
            int(rel_slice) for rel_slice in twixObj.hdr['Config']['relSliceNumber'].split()]
        slice_order = np.argsort(rel_slice_order[:num_slices])
        raw_kspace = np.asarray([raw_kspace[..., i]
                                for i in slice_order[::-1]])
        # move the slice axis to last
        raw_kspace = np.moveaxis(raw_kspace, 0, -1)
        shifts = (
            twixObj.search_header_for_val(
                'Phoenix', ('sWiPMemBlock', 'adFree', '7'))[0],
            twixObj.search_header_for_val(
                'Phoenix', ('sWiPMemBlock', 'adFree', '6'))[0],
        )
    else:
        shifts = (
            twixObj.search_header_for_val(
                'Phoenix', ('sWiPMemBlock', 'adFree', '7'))[0],
            twixObj.search_header_for_val(
                'Phoenix', ('sWiPMemBlock', 'adFree', '6'))[0],
            twixObj.search_header_for_val(
                'Phoenix', ('sWiPMemBlock', 'adFree', '8'))[0],
        )
    return raw_kspace, shifts


def phase_correction(kspace_loc, kspace_data, shifts):
    if tf.math.reduce_sum(shifts) == 0:
        return kspace_data
    phi = tf.zeros((tf.shape(kspace_data)[0], *tf.shape(kspace_data)[2:]))
    for i in range(kspace_loc.shape[1]):
        phi += kspace_loc[:, i] / np.pi / 2 * shifts[:, i][:, None]
    phase = tf.exp(-2 * np.pi * 1j * tf.cast(phi, tf.complex64))
    new_kspace_data = kspace_data * phase[:, None, :]
    return new_kspace_data


def virtual_coil_reconstruction(imgs):
    """
    Calculate the combination of all coils using virtual coil reconstruction

    Parameters
    ----------
    imgs: np.ndarray
        The images reconstructed channel by channel
        in shape [batch_size, Nch, Nx, Ny, Nz]

    Returns
    -------
    img_comb: np.ndarray
        The combination of all the channels in a complex valued
        in shape [batch_size, Nx, Ny]
    """
    img_sh = imgs.shape
    dimension = len(img_sh)-2
    # Compute first the virtual coil
    weights = tf.math.reduce_sum(tf.abs(imgs), axis=1) + 1e-16
    phase_reference = tf.cast(
        tf.math.angle(tf.math.reduce_sum(
            imgs,
            axis=(2+np.arange(len(img_sh)-2))
        )),
        tf.complex64
    )
    expand = [..., *((None, ) * (len(img_sh)-2))]
    reference = imgs / tf.cast(weights[:, None, ...], tf.complex64) / \
        tf.math.exp(1j * phase_reference)[expand]
    virtual_coil = tf.math.reduce_sum(reference, axis=1)
    difference_original_vs_virtual = tf.math.conj(imgs) * virtual_coil[:, None]
    # Hanning filtering in readout and phase direction
    hanning = tf.signal.hann_window(img_sh[-dimension])
    for d in range(dimension-1):
        hanning = tf.expand_dims(hanning, axis=-1) * tf.signal.hann_window(img_sh[dimension + d])
    hanning = tf.cast(hanning, tf.complex64)
    # Removing the background noise via low pass filtering
    if dimension == 3:    
        difference_original_vs_virtual = tf.signal.ifft3d(
            tf.signal.fft3d(difference_original_vs_virtual) * tf.signal.fftshift(hanning)
        )
    else:
        difference_original_vs_virtual = ortho_ifft2d(
            ortho_fft2d(difference_original_vs_virtual) * hanning
        )
    img_comb = tf.math.reduce_sum(
        imgs *
        tf.math.exp(
            1j * tf.cast(tf.math.angle(difference_original_vs_virtual), tf.complex64)),
        axis=1
    )
    return img_comb


def train_val_split_n_batch(dataset, split, shuffle=-1, batch_size=1, prefetch=True, len_data=None):
    if dataset.cardinality() == tf.data.UNKNOWN_CARDINALITY:
        if len_data is None:
            warnings.warn(
                'Need cardinality of data as input to split to train and val, assuming all data is given')
    else:
        len_data = len(dataset)
    if split is None:
        split = (len_data - batch_size / len_data) 
    if batch_size != 1 and ((split != 1 and batch_size > np.ceil((1 - split) * len_data)) or batch_size > np.ceil(split * len_data)):
        warnings.warn(
            'The batch size is too high, repeating the data and batching')
        # We do this as the batch size is used in setting up the network
        repeat_times = np.ceil(
            max(batch_size / (1-split), batch_size / split) / len_data)
        dataset = dataset.repeat(count=repeat_times)
        len_data = repeat_times * len_data
    if split != 1:
        train_set = dataset.take(min(np.ceil(split * len_data), len_data-1))
        val_set = dataset.skip(min(np.ceil(split * len_data), len_data-1))
    else:
        train_set = dataset
        val_set = dataset
    if shuffle is not None and shuffle:
        if shuffle < 0:
            train_set = train_set.shuffle(len(dataset))
        else:
            train_set = train_set.shuffle(shuffle)
    if batch_size:
        train_set = train_set.batch(batch_size)
        val_set = val_set.batch(batch_size)
    if prefetch:
        train_set = train_set.prefetch(1)
        val_set = val_set.prefetch(1)
    if split != 1:
        return train_set, val_set
    return train_set


def shuffle_batch_prefetch(dataset, prefetch=1, shuffle=0, seed=0, batch_size=1, unbatch=False):
    """Shuffle and batch the dataset with prefetch

    Parameters
    ----------
    datset: tf.data.Dataset
        The dataset to shuffle and batch
    prefetch: int
        The number of batches to prefetch
    shuffle: int
        The number of batches to shuffle
    batch_size: int
        The batch size
    """
    if unbatch:
        dataset = dataset.unbatch()
    if shuffle:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(prefetch)
    return dataset

