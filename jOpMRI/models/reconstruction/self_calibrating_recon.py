import numpy as np


def reconstruct_retrospective(data, mus=np.logspace(-4, 1, 5), return_all=False, n_jobs=1):
    from mri.operators import NonCartesianFFT, WaveletN
    from mri.operators.utils import normalize_frequency_locations
    from mri.reconstructors import SelfCalibrationReconstructor
    from mri.scripts.gridsearch import launch_grid

    from modopt.math.metrics import ssim, psnr
    from modopt.opt.linear import Identity
    from modopt.opt.proximity import SparseThreshold

    def _metric_adaptive_mask(test, ref, metric):
        if metric == 'ssim':
            return ssim(test, ref, ref > np.mean(ref)*0.7)
        elif metric == 'psnr':
            return psnr(test, ref, ref > np.mean(ref)*0.7)
    inputs, ref_image = data
    ref_image = ref_image[0, ..., 0]
    kspace_data, kspace_loc, smaps, extra_args = inputs
    kspace_data = kspace_data[0, ..., 0]
    kspace_loc = kspace_loc[0].T
    dcomp = extra_args[1][0]
    smaps = smaps[0]
    kspace_loc = normalize_frequency_locations(kspace_loc, Kmax=np.pi)
    metrics = {
        'ssim': {
            'metric': _metric_adaptive_mask,
            'mapping': {'x_new': 'test', 'y_new': None},
            'cst_kwargs': {'ref': ref_image, 'metric': 'ssim'},
            'early_stopping': True,
        },
        'psnr': {
            'metric': _metric_adaptive_mask,
            'mapping': {'x_new': 'test', 'y_new': None},
            'cst_kwargs': {'ref': ref_image, 'metric': 'psnr'},
            'early_stopping': True,
        },
    }
    linear_params = {
        'init_class': WaveletN,
        'kwargs':
            {
                'wavelet_name': 'sym8',
                'nb_scale': 4,
            }
    }
    regularizer_params = {
        'init_class': SparseThreshold,
        'kwargs':
            {
                'linear': Identity,
                'weights': mus,
            }
    }
    optimizer_params = {
        'kwargs':
            {
                'optimization_alg': 'fista',
                'num_iterations': 15,
                'metrics': metrics,
                'metric_call_period': 5,
            }
    }
    fourier_op = NonCartesianFFT(
        kspace_loc,
        implementation='gpuNUFFT',
        shape=ref_image.shape,
        density_comp=np.abs(dcomp),
        n_coils=kspace_data.shape[0],
        smaps=smaps,
    )
    results = launch_grid(
        kspace_data=kspace_data,
        fourier_op=fourier_op,
        linear_params=linear_params,
        regularizer_params=regularizer_params,
        optimizer_params=optimizer_params,
        reconstructor_class=SelfCalibrationReconstructor,
        reconstructor_kwargs={
            'gradient_formulation': 'synthesis'
        },
        compare_metric_details={'metric': 'ssim'},
        n_jobs=n_jobs,
        verbose=15,
    )
    if return_all:
        return results
    else:
        return results[0][results[-1]][0], results[0][results[-1]][-1], mus[results[-1]], ref_image
