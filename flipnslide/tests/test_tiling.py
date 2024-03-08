''' Test tiling.py '''

import pytest
import numpy as np
import torch
from ..tiling import FlipnSlide, Tiling


tile_size = 256

@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.random.rand(3, 5120, 5120) 


def test_flipnslide(sample_image):
    '''This also tests `sliding_transforms` method on `Tiling` class.'''
    
    # Test tiling with the flipnslide strategy
    flipnslide = FlipnSlide(tile_size=tile_size, 
                            data_type='array', 
                            image=sample_image)
    
    # Test that a numpy array is returned and that it is the correct dims
    assert isinstance(flipnslide.tiles, np.ndarray)
    assert len(flipnslide.tiles.shape) == 4
    
    # Test that the correct tile shape was created
    assert flipnslide.tiles.shape[-1] == flipnslide.tiles.shape[-2] == tile_size
    
    # Test that the right amount of tiles were created
    assert flipnslide.tiles.shape[0] == 2890
    
    # Test that permute_idx was created and is correct length
    assert len(flipnslide.tiles) == len(flipnslide.permute_idx)
    

def test_tiling(sample_image):

    # Test no_overlap method
    no_overlap = Tiling(tile_size=tile_size,
                        tile_style='no_overlap',
                        data_type='array', 
                        image=sample_image)
    
    # Test that a numpy array is returned and that it is the correct dims
    assert isinstance(no_overlap.tiles, np.ndarray)
    assert len(no_overlap.tiles.shape) == 4
    
    # Test that the correct tile shape was created
    assert no_overlap.tiles.shape[-1] == no_overlap.tiles.shape[-2] == tile_size
    
    # Test that the right amount of tiles were created
    assert no_overlap.tiles.shape[0] == 400

    # Test overlap method
    overlap = Tiling(tile_size=tile_size,
                     tile_style='overlap',
                     data_type='array', 
                     image=sample_image)
    
    # Test that a numpy array is returned and that it is the correct dims
    assert isinstance(no_overlap.tiles, np.ndarray)
    assert len(overlap.tiles.shape) == 4
    
    # Test that the correct tile shape was created
    assert overlap.tiles.shape[-1] == overlap.tiles.shape[-2] == tile_size
    
    # Test that the right amount of tiles were created
    assert overlap.tiles.shape[0] == 1521
    
    
def test_crop():
    
    # Initialize sample image of incorrect size
    sample_image = np.random.rand(3, 1000, 1023)
    
    #Test that the crop function works
    tiling_init = Tiling(image=sample_image)
    cropped_image = tiling_init.crop(sample_image, tile_size=tile_size)
    
    # Test that the image is cropped
    assert cropped_image.shape[-1] < sample_image.shape[-1]
    assert cropped_image.shape[-2] < sample_image.shape[-2]
    
    # Test that the image is square
    assert cropped_image.shape[-1] == cropped_image.shape[-2]
    
    # Test that the image is cropped to a size that fits the tile_size
    assert cropped_image.shape[-1] % tile_size == 0
    
    
def test_torch(sample_image):
    
    # Initialize tiles
    no_overlap = Tiling(tile_size=tile_size,
                        tile_style='no_overlap',
                        data_type='tensor', 
                        image=sample_image)
    
    # Test that tiles are a Torch tensor
    assert isinstance(no_overlap.tiles, torch.Tensor)
    

if __name__ == '__main__':
    pytest.main()