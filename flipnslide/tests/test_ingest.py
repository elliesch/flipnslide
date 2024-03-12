''' Test ingest.py '''

import pytest
import numpy as np
from ..ingest import ImageIngest



def test_imageingest():
    
    # Initial conditions to test
    coords = [40.730610, 40.850610, -73.865242, -74.031297]
    time_range = '2023-01-31/2023-02-28'
    
    image_ingest = ImageIngest(coords, time_range)
    
    # Test that coords and time_range are attributes
    assert image_ingest.coords == coords
    assert image_ingest.time_range == time_range
    
    # Test that an image is downloaded
    assert isinstance(image_ingest.image, np.ndarray)
    
    # Test that the image has the expected number of dims
    assert len(image_ingest.image.shape) == 3
    

if __name__ == '__main__':
    pytest.main()