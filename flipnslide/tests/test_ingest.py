''' Test ingest.py '''

import pytest
from .ingest import ImageIngest



# Test case for initializing ImageIngest object
def test_imageingest():
    
    coords = [36.473972, 39.073972, -120.831297, -124.031297]
    time_range = '2020-01-31/2021-01-31'
    
    image_ingest = ImageIngest(coords, time_range)
    
    # Test that coords and time_range are attributes
    assert image_ingest.coords == coords
    assert image_ingest.time_range == time_range
    
    # Test that an image is downloaded
    assert isinstance(image_ingest.image, np.ndarray)
    

if __name__ == '__main__':
    pytest.main()