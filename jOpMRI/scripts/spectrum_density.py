import hydra
import numpy as np
import os

from jOpMRI.data.spectrum import SpectrumBuilder
from jOpMRI.config import FASTMRI_DATA_DIR
from jOpMRI.models.acquisition.sampling_density import get_spectrum_based_density


@hydra.main(config_path='configs', config_name='spectrum_density')
def get_sampling_densities(cfg):
    spectrum_dataset = SpectrumBuilder(
        path=os.path.join(FASTMRI_DATA_DIR, cfg['data']['train_path']),
        multicoil=cfg['data']['coils'] == 'multicoil',
        return_images=False,
        repeat=False,
        split_slices=True,
        contrast=cfg['data']['contrast'],
        n_samples=cfg['data']['n_samples'],
        scale_factor=1e6,
    )
    densities = get_spectrum_based_density(spectrum_dataset)
    np.save(os.path.join(os.getcwd(), 'spectrum_based'), densities)


if __name__ == '__main__':
    get_sampling_densities()
