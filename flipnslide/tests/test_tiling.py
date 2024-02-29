''' Test tiling.py '''

import pytest
import numpy as np
from ..tiling import FlipnSlide, Tiling



@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.random.rand(3, 5120, 5120)  

def test_flipnslide(sample_image):
    # Test tiling with the flipnslide strategy
    flipnslide = FlipnSlide(tile_size=256, data_type='array')
    tiles = flipnslide.tiles(sample_image)

    assert isinstance(tiles, np.ndarray)
    # need to figure out this number
    # assert tiles.shape[0] == ?
    

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