''' Test util.py '''

import pytest
import numpy as np
from ..util import download_image, preprocess


def test_download_image():
    
    # Initial conditions to test
    coords = [40.730610, 40.850610, -73.865242, -74.031297]
    time_range = '2023-01-31/2023-02-28'
    
    downloaded_image = download_image(coords, time_range)
    
    # Test that an image is downloaded
    assert isinstance(downloaded_image, np.ndarray)
    
    # Test that the image has the expected number of dims
    assert len(downloaded_image.shape) == 3

    
def test_preprocess():
    
    # Create downloaded image to test
    # Initial conditions to test
    coords = [40.730610, 40.850610, -73.865242, -74.031297]
    time_range = '2023-01-31/2023-02-28'
    
    downloaded_image = download_image(coords, time_range)
    preprocessed_image = preprocess(downloaded_image)
    
    # Test that the image is still the right type
    assert isinstance(preprocessed_image, np.ndarray)
    
    # Test that preprocessing retained correct dims
    assert len(preprocessed_image.shape) == len(downloaded_image.shape)
    
    # Test that preprocessing removed nans
    assert not np.isnan(preprocessed_image).any(), "Array contains NaN values"
    
    # Test that the preprocessed array is centered on zero
    mean = np.mean(preprocessed_image)
    tolerance = 1e-2

    # Check if the mean is close to zero
    assert abs(mean) < tolerance, f"Mean ({mean}) is not close to zero within tolerance {tolerance}"


if __name__ == '__main__':
    pytest.main()