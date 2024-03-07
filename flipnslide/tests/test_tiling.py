''' Test tiling.py '''

import pytest
import numpy as np
from ..tiling import FlipnSlide, Tiling



@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.random.rand(3, 5120, 5120) 


tile_size = 256


def test_flipnslide(sample_image):
    # Test tiling with the flipnslide strategy
    flipnslide = FlipnSlide(tile_size=tile_size, 
                            data_type='array', 
                            image=sample_image)
    
    # Test that a numpy array is returned and that it is the correct dims
    assert isinstance(flipnslide.tiles, np.ndarray)
    assert len(flipnslide.tiles.shape) == 4
    
    # Test that the right amount of tiles were created
    assert tiles.shape[0] == 2890
    

def test_tiling(sample_image):
    tile_size = 256

    # Test no_slide_tile method
    tiles_no_slide = Tiling.no_slide_tile(sample_image, tile_size)
    assert isinstance(tiles_no_slide, np.ndarray)
    # need to figure out this number
    # assert tiles.shape[0] == ?

    # Test sliding_tile method
    tiles_sliding = Tiling.sliding_tile(sample_image, tile_size)
    assert isinstance(tiles_sliding, np.ndarray)
    # need to figure out this number
    # assert tiles.shape[0] == ?

    # Test sliding_transforms method
    tiles_flipnslide = Tiling.sliding_transforms(sample_image, tile_size)
    assert isinstance(tiles_flipnslide, np.ndarray)
    # need to figure out this number
    # assert tiles.shape[0] == ?


if __name__ == '__main__':
    pytest.main()