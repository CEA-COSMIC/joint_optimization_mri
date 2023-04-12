import tensorflow as tf


def get_spectrum_based_density(spectrum_dataset):
    """
    Function to obtain density directly based on spectrum of the dataset
    This method works by simple averaging across spectrums from various
    MRI data.
    :param dataset: The dataset as a tf dataset.
    :return: The average spectrums: both log and natural
    """
    spectrum = spectrum_dataset._preprocessed_ds
    num_data = len(spectrum)
    if spectrum_dataset.log_spectrum:
        total_spectrums = spectrum.reduce(
            (tf.zeros((320, 320), dtype='float32'),
             tf.zeros((320, 320), dtype='float32')),
            lambda run_sum, data:
            (run_sum[0] + tf.abs(data[0][0][..., 0] / num_data),
             run_sum[1] + data[1][0][..., 0] / num_data)
        )
    else:
        total_spectrums = spectrum.reduce(
            tf.zeros((320, 320), dtype='float32'),
            lambda run_sum, data: (
                run_sum + tf.abs(data[0][..., 0]) / num_data,)
        )
    densities = []
    for total_spectrum in total_spectrums:
        # Move the density to be >0, this is useful for log spectrum
        density = total_spectrum - tf.math.reduce_min(total_spectrum)
        # Normalize the density
        densities.append(density / tf.math.reduce_sum(density))
    densities = tf.stack(densities)
    return densities
