''' Flip-n-Slide Tiling and Permutations -- Core Package Functionality '''

import numpy as np
import torch
from .tiling import ImageIngest
from .util import saver

class Tiling:
    
    def __init__(self, tile_size:int=256,
                 tile_style:str = 'flipnslide',
                 data_type:str = 'tensor',
                 save:bool = False,
                 **kwargs):
        '''
        Initialize Tiling with the given parameters.

        Example:
        Initialize the Tiling class to obtain a tensor of tiles sized 256 x 256:
        >>> tiles = Tiling(image, tile_style='flipnslide')

        Parameters:
        - tile_size (int): Integer representing the size of the tile (default is 256).
        - tile_style (str): String representing the tiling method, should be one of ['flipnslide', 'overlap', 'no_overlap'] (default is 'flipnslide').
        - data_type (str): String representing output data type, should be one of ['tensor', 'array'], where 'tensor' is a PyTorch tensor and 'array' is a NumPy ndarray (default is 'tensor').
        - save (bool): Boolean indicating whether to save the file to local memory (default is False).

        Scientific Image Parameters:

        Required Parameter for Use with PreDownloaded Image:
        - image (numpy.ndarray): NumPy ndarray representing the large input image with one dimension for channels, one dimension for x pixels, and one dimension for y pixels.

        OR

        Required Parameters for Downloading Image:
        - coords (List[float]): List of four floats indicating corners of the requested image in long/lat coordinates. Should follow this format: [southern_boundary, northern_boundary, eastern_boundary, western_boundary].
        - time_range (str): String indicating the time range for the requested image. Should follow this format: 'YYYY-MM-DD/YYYY-MM-DD'.

        Optional Parameters for Downloading Image:
        - bands (List[str]): List indicating bands of the requested image (default is ['blue', 'green', 'red', 'nir08']).
        - cat_name (List[str]): List indicating requested catalogs to query in Planetary Computer (default is ['landsat-c2-l2']).
        - cloud_cov (int): Integer representing the maximum percentage of cloud cover for the requested image (default is 5).
        - res (int): Integer representing the resolution of the requested image. Should match the resolution of the data catalog (default is 30).

        Raises:
        - ValueError: Raised if an invalid tile_style is provided. Allowed values are 'flipnslide', 'overlap', or 'no_overlap'.
        - AssertionError: Raised if input image is not a NumPy array or if 'coords' are not provided as a list of four floats.

        Notes:
        - The tiles are generated based on the specified tiling method (tile_style) and are returned as either a PyTorch tensor or a NumPy ndarray based on the data_type parameter.
        - If 'save' is set to True, the generated tiles will be saved to local memory.
        '''
        
        # Tunable params
        self.tile_size = tile_size
        
        # Image or Labels to be tiled
        if 'image' in kwargs:
            
            # Make sure that downloaded image is a numpy array
            assert isinstance(kwargs['image'], np.ndarray), "Input image is not a NumPy array."
            
            image = kwargs['image']
            
        else:
            
            if 'coords' and 'time_range' in kwargs:
                
                assert len(coords) == 4 and all(isinstance(x, float) for x in coords), "'coords' should be a list of four floats."
                assert isinstance(kwargs['time_range'], str), "'time_range' should be a string of the format 'YYYY-MM-DD/YYYY-MM-DD'."
                
                coords = kwargs.pop('coords')
                time_range = kwargs.pop('time_range')
                
            else:
                raise ValueError("Either an 'image' or ('coords' and 'time_range') keyword arguments need to be provided.")
        
            image = ImageIngest(coords, time_range, **kwargs).image
            
        # Tiling method to be implemented
        if tile_style not in ['flipnslide', 'overlap', 'no_overlap']:
            raise ValueError("Invalid style. Allowed values are 'flipnslide', 'overlap', or 'no_overlap'.")
        
        if data_type == 'tensor':
            if tile_style == 'flipnslide':            
                self.tiles = torch.from_numpy(sliding_transforms(image, self.tile_size))
            elif tile_style == 'overlap':
                self.tiles = torch.from_numpy(sliding_tile(image, self.tile_size))
            else:
                self.tiles = torch.from_numpy(no_slide_tile(image, self.tile_size))
            
        elif data_type == 'array':
            if tile_style == 'flipnslide':            
                self.tiles = sliding_transforms(image, self.tile_size)
            elif tile_style == 'overlap':
                self.tiles = sliding_tile(image, self.tile_size)
            else:
                self.tiles = no_slide_tile(image, self.tile_size)
            
        # Save the data
        if save == True:
            saver(self.tiles, file_type=data_type, file_name=f'{tile_style}_tiles')      
            

    def no_slide_tile(image, tile_size):
        '''
        Divide an image into non-overlapping tiles of specified size.

        Parameters:
        - image (numpy.ndarray): The input image to be divided into tiles.
        - tile_size (int): The size of the tiles. The image will be divided into tiles of size
          `tile_size x tile_size`.

        Returns:
        - numpy.ndarray: An array containing the tiles of the input image. The shape of the
          returned array is `(num_tiles, num_channels, tile_size, tile_size)`, where `num_tiles`
          is the total number of tiles and `num_channels` is the number of color channels in
          the input image.
        '''

        #find the tile indices
        shape = image.shape
        side = max(shape)
        n_channels = min(shape)
        count_1d = int(side/tile_size)

        #initialize the tile arrays
        image_tiles = np.empty([count_1d, count_1d, n_channels, tile_size, tile_size])

        #fold into tiles
        fold_idx = np.arange(0, side, tile_size)

        for idx_x in range(len(fold_idx)-1):
            for idx_y in range(len(fold_idx)-1):

                #tile the images
                image_tile = image[:, fold_idx[idx_x]:fold_idx[idx_x+1], fold_idx[idx_y]:fold_idx[idx_y+1]]
                image_tiles[idx_x, idx_y, :, :, :] = image_tile

        #define the tiles
        image_tiles = image_tiles.reshape(-1, n_channels, tile_size, tile_size)

        return image_tiles 
    
    def sliding_tile(image, tile_size):
        '''
        Divide an image into overlapping tiles of specified size with a stride of half the tile size.

        Parameters:
        - image (numpy.ndarray): The input image to be divided into tiles.
        - tile_size (int): The size of the tiles. The image will be divided into tiles of size
          `tile_size x tile_size`.

        Returns:
        - numpy.ndarray: An array containing the tiles of the input image. The shape of the
          returned array is `(num_tiles, num_channels, tile_size, tile_size)`, where `num_tiles`
          is the total number of tiles and `num_channels` is the number of color channels in
          the input image.

        Notes:
        - The tiles are generated with a sliding window approach, resulting in overlapping tiles
          with a stride of half the tile size. This means that each pixel in the input image
          will appear in multiple tiles, except for the pixels along the borders.
        '''
    
        #find the tile indices
        shape = image.shape
        side = max(shape)
        n_channels = min(shape)
        count_1d = int(side/tile_size) + (int(side/tile_size) - 1)

        #initialize the tile arrays
        image_tiles = np.empty([count_1d, count_1d, n_channels, tile_size, tile_size])

        #fold into tiles
        fold_idx = np.arange(0, side, int(tile_size/2))

        for idx_x in range(len(fold_idx)-1):
            for idx_y in range(len(fold_idx)-1):

                #tile the images
                image_tile = image[:, fold_idx[idx_x]:(fold_idx[idx_x]+tile_size), 
                                      fold_idx[idx_y]:(fold_idx[idx_y]+tile_size)]
                image_tiles[idx_x, idx_y, :, :, :] = image_tile

        #define the tiles
        image_tiles = image_tiles.reshape(-1, n_channels, tile_size, tile_size)

        return image_tiles
    
    def sliding_transforms(image, tile_size):
        '''
        Divide an image into overlapping tiles with rotational and flip augmentations applied to each tile.

        Parameters:
        - image (numpy.ndarray): The input image to be divided into tiles.
        - tile_size (int): The size of the tiles. The image will be divided into tiles of size
          `tile_size x tile_size`.

        Returns:
        - numpy.ndarray: An array containing the tiles of the input image with augmentations applied.
          The shape of the returned array is `(num_tiles, num_channels, tile_size, tile_size)`, where
          `num_tiles` is the total number of tiles and `num_channels` is the number of color channels
          in the input image.

        Notes:
        - The tiles are generated with a sliding window approach, resulting in overlapping tiles
          with a stride of half, one quarter, and three quarters the tile size. This means that each 
          pixel in the input image will appear in multiple tiles, except for the pixels along the borders.
        - Rotational and flip augmentations are applied to each tile based on its position.
        - Augmentations include horizontal flips, vertical flips, and rotations of 90, 180, and 270 degrees.
          Distinct augmentations are applied to each tile that overlap with another to eliminate
          redundancies between overlapping pixels.
        '''
    
        #find the tile indices
        shape = image.shape
        side = max(shape)
        n_channels = min(shape)
        count_1d = int(side/tile_size) + (int(side/tile_size) - 1)

        #initialize the tile arrays
        image_tiles = np.empty([count_1d, count_1d, n_channels, tile_size, tile_size])
        idx_tiles = []

        #fold into tiles
        fold_idx = np.arange(0, side, int(tile_size/2))

        for idx_x in range(len(fold_idx)-1):
            for idx_y in range(len(fold_idx)-1):

                #tile the images
                image_tile = image[:, fold_idx[idx_x]:(fold_idx[idx_x]+tile_size), 
                                      fold_idx[idx_y]:(fold_idx[idx_y]+tile_size)]

                #add rotational augmentations where needed
                #where both are divisible by two, no rotations happen
                if (idx_x % 2 != 0) & (idx_y % 2 != 0):
                    image_tile = np.rot90(image_tile, k=3, axes=(1,2)) #270
                    #track the indices
                    idx_tiles.append(1)

                elif (idx_x % 2 != 0) & (idx_y % 2 == 0):
                    image_tile = np.rot90(image_tile, k=2, axes=(1,2)) #180
                    #track the indices
                    idx_tiles.append(2)

                elif (idx_x % 2 == 0) & (idx_y % 2 != 0):
                    image_tile = np.rot90(image_tile, k=1, axes=(1,2)) #90
                    #track the indices
                    idx_tiles.append(3)

                else:
                    #track the indices
                    idx_tiles.append(0)

                image_tiles[idx_x, idx_y, :, :, :] = image_tile

        #define the tiles
        image_tiles = image_tiles.reshape(-1, n_channels, tile_size, tile_size)

        ###======== TILL HERE THE CODE IS SLIDING TILES WITH ROTATION AUGMENTATION ADDED ========###
        ###======== FOLLOWING IS INNER TILES WITH FLIP + ROTATION AUGMENTATION ADDED ========###

        #find the tile indices for the 25/75% slide
        inner_image = image[:, 64:-64, 64:-64]
        shape = inner_image.shape
        side = max(shape)
        count_1d = int(side/tile_size) + (int(side/tile_size) - 1)

        #initialize the tile arrays and starting point for index tracking
        inner_image_tiles = np.empty([count_1d, count_1d, n_channels, tile_size, tile_size])
        # idx_starting_point = max(max(idx_tiles)) + 1 #since there will be a zero

        #fold into tiles
        fold_idx = np.arange(0, side, int(tile_size/2))

        if tile_size == 64:
            adjuster = 1
        elif tile_size == 128:
            adjuster = 1
        elif tile_size == 256:
            adjuster = 2
        elif tile_size == 512:
            adjuster = 3

        for idx_x in range(len(fold_idx)-adjuster):
            for idx_y in range(len(fold_idx)-adjuster):

                #tile the images
                image_tile = inner_image[:, fold_idx[idx_x]:(fold_idx[idx_x]+tile_size), 
                                            fold_idx[idx_y]:(fold_idx[idx_y]+tile_size)]

                #add rotational augmentations and flip augmentations
                #0 degrees gets no rotation and only flips
                #90 degrees gets rotation and flips
                #remaining rotations + flips are redundant
                if (idx_x % 2 == 0) & (idx_y % 2 == 0):
                    image_tile = image_tile[:,:,::-1] #horizontal flip
                    #track the indices
                    idx_tiles.append(4)

                elif (idx_x % 2 == 0) & (idx_y % 2 != 0):
                    image_tile = image_tile[:,::-1,:] #vertical flip
                    #track the indices
                    idx_tiles.append(5)

                elif (idx_x % 2 != 0) & (idx_y % 2 != 0):
                    image_tile = np.rot90(image_tile, k=1, axes=(1,2))
                    image_tile = image_tile[:,:,::-1] 
                    #track the indices
                    idx_tiles.append(6)

                elif (idx_x % 2 != 0) & (idx_y % 2 == 0):
                    image_tile = np.rot90(image_tile, k=1, axes=(1,2))
                    image_tile = image_tile[:,::-1,:] 
                    #track the indices
                    idx_tiles.append(7)

                inner_image_tiles[idx_x, idx_y, :, :, :] = image_tile

        #define the tiles
        inner_image_tiles = inner_image_tiles.reshape(-1, n_channels, tile_size, tile_size)

        ## Combine all the tiles
        all_image_tiles = np.concatenate((image_tiles, inner_image_tiles), axis=0)

        return all_image_tiles