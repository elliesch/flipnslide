''' Tiling and Permutations -- Central Functionality '''

import numpy as np
from .tiling import ImageIngest

class FlipnSlide:
    
    
class Tiling:
    
    def __init__(self, tile_size,
                 tile_style = 'flipnslide',
                 **kwargs):
        
        # Tunable params
        self.tile_size = tile_size
        
        # Image or Labels to be tiled
        if 'image' in kwargs:
            
            # Make sure that downloaded image is a numpy array
            assert isinstance(kwargs['image'], np.ndarray), "Input image is not a NumPy array."
            
            image = kwargs['image']
            
        else:
            #make code to import image through .ingest
            
            

    def no_slide_tile(channels, masks, tile_size):

        #find the tile indices
        shape = channels.shape
        side = max(shape)
        n_channels = min(shape)
        n_labels = min(masks.shape)
        count_1d = int(side/tile_size)

        #initialize the tile arrays
        image_tiles = np.empty([count_1d, count_1d, n_channels, tile_size, tile_size])
        label_tiles = np.empty([count_1d, count_1d, n_labels, tile_size, tile_size])

        #fold into tiles
        fold_idx = np.arange(0, side, tile_size)

        for idx_x in range(len(fold_idx)-1):
            for idx_y in range(len(fold_idx)-1):

                #tile the images
                image_tile = channels[:, fold_idx[idx_x]:fold_idx[idx_x+1], fold_idx[idx_y]:fold_idx[idx_y+1]]
                image_tiles[idx_x, idx_y, :, :, :] = image_tile

                #tile the labels
                label_tile = masks[:, fold_idx[idx_x]:fold_idx[idx_x+1], fold_idx[idx_y]:fold_idx[idx_y+1]]
                label_tiles[idx_x, idx_y, :, :, :] = label_tile

        #define the tiles
        image_tiles = image_tiles.reshape(-1, n_channels, tile_size, tile_size)
        label_tiles = label_tiles.reshape(-1, n_labels, tile_size, tile_size)

        return image_tiles, label_tiles    