''' Test ingest.py '''

import pytest
import numpy as np
from ..viz import ingest_viz, crop_viz, tile_viz



@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.random.rand(3, 256, 300)  


@pytest.fixture
def sample_crop():
    # Create a sample cropped image for testing
    return np.random.rand(3, 128, 128)  


@pytest.fixture
def sample_tiles():
    # Create a sample array of tiles for testing
    return np.random.rand(10, 3, 64, 64) 


def test_ingest_viz(sample_image):
    # Test the ingest_viz function
    ingest_viz(sample_image)
    

def test_crop_viz(sample_image, sample_crop):
    # Test the crop_viz function
    crop_viz(sample_image, sample_crop)
    

def test_tile_viz(sample_tiles):
    # Test the tile_viz function
    tile_viz(sample_tiles)
    
if __name__ == '__main__':
    pytest.main()