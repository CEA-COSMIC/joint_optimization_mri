import pytest

from jOpMRI.data.images import ImagesBuilder
from jOpMRI.data.spectrum import SpectrumBuilder, JustCropScaleData

from tf_fastmri_data.test_utils import key_map


@pytest.mark.parametrize('multicoil', [True, False])
@pytest.mark.parametrize('crop_image_data', [True, False])
def test_just_crop_dataset(multicoil, crop_image_data, create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset[key_map[multicoil]['train']]
    JustCropScaleData(path=path, multicoil=multicoil, crop_image_data=crop_image_data)


@pytest.mark.parametrize('multicoil', [True, False])
@pytest.mark.parametrize('log_spectrum', [True, False])
def test_spectrum_builder(multicoil, log_spectrum, create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset[key_map[multicoil]['train']]
    SpectrumBuilder(path=path, multicoil=multicoil, log_spectrum=log_spectrum)


@pytest.mark.parametrize('multicoil', [True, False])
@pytest.mark.parametrize('crop_image_data', [True, False])
def test_image_dataset_builder(multicoil, crop_image_data, create_full_fastmri_test_tmp_dataset):
    path = create_full_fastmri_test_tmp_dataset[key_map[multicoil]['train']]
    ImagesBuilder(path=path, multicoil=multicoil, crop_image_data=crop_image_data)