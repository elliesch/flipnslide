''' Flip-n-Slide Tiling and Permutations -- Core Package Functionality '''

import numpy as np
import torch
import tensorflow as tf
from .ingest import ImageIngest
from .util import saver
from .viz import ingest_viz, crop_viz, tile_viz



class FlipnSlide:
    
    def __init__(self, tile_size:int=256,
                 data_type:str = 'tensor',
                 save:bool = False,
                 viz:bool = False,
                 verbose:bool = False,
                 **kwargs):
        '''
        Initialize abbreviated tiling class that only uses Flip-n-Slide tiling strategy.
        
        Example:
        Initialize the FlipnSlide class to obtain a tensor of tiles sized 256 x 256:
        >>> tiles = FlipnSlide(image).tiles
        
        Attributes:
        - tiles (numpy.ndarray OR torch.tensor): PyTorch tensor or a NumPy ndarray. The shape of the
          returned array is `(num_tiles, num_channels, tile_size, tile_size)`, where `num_tiles`
          is the total number of tiles and `num_channels` is the number of color channels in
          the input image.
        - tile_size (int): Integer representing the size of the tile side.

        Tiling Parameters:
        - tile_size (int): Integer representing the size of the tile (default is 256).
        - tile_style (str): String representing the tiling method, should be one of 
          ['flipnslide', 'overlap', 'no_overlap'] (default is 'flipnslide').
        - data_type (str): String representing output data type, should be one of ['tensor', 'array'], 
          where 'tensor' is a PyTorch tensor and 'array' is a NumPy ndarray (default is 'tensor').
        - save (bool): Boolean indicating whether to save the file to local memory (default is False).
        - viz (bool): Boolean indicating whether to show visualizations of image and 
          tiles (default is False).
        - verbose (bool): Boolean indicating whether to print stages of tiling (default is False).

        Scientific Image Parameters:

        Required Parameter for Use with PreDownloaded Image:
        - image (numpy.ndarray): NumPy ndarray representing the large input image. The dimensions must 
          be in the following order (n_channels, n_pix, n_pix). This release will reprocess the image 
          to be a square that is divisible by the tile size.

        OR

        Required Parameters for Downloading Image:
        - coords (List[float]): List of four floats indicating corners of the requested image in 
          long/lat coordinates. Should follow this format: 
          [southern_boundary, northern_boundary, eastern_boundary, western_boundary].
        - time_range (str): String indicating the time range for the requested image. 
          Should follow this format: 'YYYY-MM-DD/YYYY-MM-DD'.

        Optional Parameters for Downloading Image:
        - bands (List[str]): List indicating bands of the requested image 
          (default is ['blue', 'green', 'red', 'nir08']).
        - cat_name (List[str]): List indicating requested catalogs to query in Planetary Computer 
          (default is ['landsat-c2-l2']).
        - cloud_cov (int): Integer representing the maximum percentage of cloud cover for the 
          requested image (default is 5).
        - res (int): Integer representing the resolution of the requested image. Should match the 
          resolution of the data catalog (default is 30).

        Raises:
        - ValueError: Raised if an invalid tile_style is provided. 
          Allowed values are 'flipnslide', 'overlap', or 'no_overlap'.
        - AssertionError: Raised if input image is not a NumPy array or if 'coords' are not provided 
          as a list of four floats.
        '''
        
        self.tile_size = tile_size
        
        init_tiling = Tiling(tile_size=tile_size, tile_style='flipnslide',
                             data_type=data_type, save=save, 
                             viz=viz, verbose=verbose, **kwargs)
        
        self.tiles = init_tiling.tiles
        
        self.permute_idx = init_tiling.permute_idx


class Tiling:
    
    def __init__(self, tile_size:int=256,
                 tile_style:str = 'flipnslide',
                 data_type:str = 'tensor',
                 save:bool = False,
                 viz:bool = False,
                 verbose:bool = False,
                 **kwargs):
        '''
        Initialize Tiling with the given parameters.

        Example:
        Initialize the Tiling class to obtain a tensor of tiles sized 256 x 256:
        >>> tiles = Tiling(image, tile_style='flipnslide').tiles
        
        Attributes:
        - tiles (numpy.ndarray OR torch.tensor): PyTorch tensor or a NumPy ndarray. The shape of the
          returned array is `(num_tiles, num_channels, tile_size, tile_size)`, where `num_tiles`
          is the total number of tiles and `num_channels` is the number of color channels in
          the input image.
        - tile_size (int): Integer representing the size of the tile side.

        Tiling Parameters:
        - tile_size (int): Integer representing the size of the tile (default is 256).
        - tile_style (str): String representing the tiling method, should be one of 
          ['flipnslide', 'overlap', 'no_overlap'] (default is 'flipnslide').
        - data_type (str): String representing output data type, should be one of ['tensor', 'array'], 
          where 'tensor' is a PyTorch tensor and 'array' is a NumPy ndarray (default is 'tensor').
        - save (bool): Boolean indicating whether to save the file to local memory (default is False).
        - viz (bool): Boolean indicating whether to show visualizations of image and 
          tiles (default is False).
        - verbose (bool): Boolean indicating whether to print stages of tiling (default is False).

        Scientific Image Parameters:

        Required Parameter for Use with PreDownloaded Image:
        - image (numpy.ndarray): NumPy ndarray representing the large input image. The dimensions must 
          be in the following order (n_channels, n_pix, n_pix). This release will reprocess the image 
          to be a square that is divisible by the tile size.

        OR

        Required Parameters for Downloading Image:
        - coords (List[float]): List of four floats indicating corners of the requested image in 
          long/lat coordinates. Should follow this format: 
          [southern_boundary, northern_boundary, eastern_boundary, western_boundary].
        - time_range (str): String indicating the time range for the requested image. 
          Should follow this format: 'YYYY-MM-DD/YYYY-MM-DD'.

        Optional Parameters for Downloading Image:
        - bands (List[str]): List indicating bands of the requested image 
          (default is ['blue', 'green', 'red', 'nir08']).
        - cat_name (List[str]): List indicating requested catalogs to query in Planetary Computer 
          (default is ['landsat-c2-l2']).
        - cloud_cov (int): Integer representing the maximum percentage of cloud cover for the 
          requested image (default is 5).
        - res (int): Integer representing the resolution of the requested image. Should match the 
          resolution of the data catalog (default is 30).

        Raises:
        - ValueError: Raised if an invalid tile_style is provided. 
          Allowed values are 'flipnslide', 'overlap', or 'no_overlap'.
        - AssertionError: Raised if input image is not a NumPy array or if 'coords' are not provided 
          as a list of four floats.

        Notes:
        - The tiles are generated based on the specified tiling method (tile_style) and are returned 
          as either a PyTorch tensor or a NumPy ndarray based on the data_type parameter.
        - If 'save' is set to True, the generated tiles will be saved to local memory.
        '''
        
        # Tunable params
        self.tile_size = tile_size
        
        # Image or Labels to be tiled
        if 'image' in kwargs:
            
            # Make sure that downloaded image is a numpy array
            assert isinstance(kwargs['image'], np.ndarray), "Input image is not a NumPy array."
            
            image = kwargs['image']
            
            if verbose == True:
                print('Provided image is being processed...')
            
        else:
            
            if 'coords' and 'time_range' in kwargs:
                
                assert (len(kwargs['coords']) == 4 
                        and all(isinstance(x, float) for x in kwargs['coords'])
                       ), "'coords' should be a list of four floats."
                assert (isinstance(kwargs['time_range'], str)
                       ), "'time_range' should be a string of the format 'YYYY-MM-DD/YYYY-MM-DD'."
                
                coords = kwargs.pop('coords')
                time_range = kwargs.pop('time_range')
                
            else:
                raise ValueError("Either an 'image' or ('coords' and 'time_range') keyword arguments need to be provided.")
                
            if verbose == True:
                print('Requested image is being downloaded via Planetary Computer...')
                
                image = ImageIngest(coords, time_range, 
                                    verbose = True, **kwargs).image
                
            else:
                image = ImageIngest(coords, time_range, **kwargs).image
        
        # Visualize imported image
        if viz == True or verbose == True:
            ingest_viz(image)
            
        # Crop image to square divisible by tile size
        if image.shape[-1] % self.tile_size != 0 or image.shape[-2] % self.tile_size != 0:
            
            if verbose == True:
                print('Image is being cropped to a square that is divisible by the tile size...')
            
            crop = self.crop(image, self.tile_size)
            
            if viz == True or verbose == True:
                crop_viz(image, crop)
                
            image = crop
            
        # Implement chosen tiling method
        if tile_style not in ['flipnslide', 'overlap', 'no_overlap']:
            raise ValueError("Invalid style. Allowed values are 'flipnslide', 'overlap', or 'no_overlap'.")
        
        if verbose == True:
                print(f'Image is being tiled using the {tile_style} approach...')
        
        if tile_style == 'flipnslide':            
            self.tiles, self.permute_idx = self.sliding_transforms(image, self.tile_size)
        elif tile_style == 'overlap':
            self.tiles = self.sliding_tile(image, self.tile_size)
        else:
            self.tiles = self.no_slide_tile(image, self.tile_size)
            
        if viz == True or verbose==True:
            tile_viz(self.tiles)
        
        # Optional Move to tensor
        if data_type not in ['array', 'tensor', 'tensor_tf']:
            raise ValueError("Invalid output type. Allowed values are 'array' for a numpy array, 'tensor' for a PyTorch tensor, or 'tensor_tf' for a tensorflow tensor.")
        
        if data_type == 'tensor':
            if verbose == True:
                print('Tiles are being converted to PyTorch tensor...')
            
            self.tiles = torch.from_numpy(self.tiles)

        elif data_type == 'tensor_tf':
            if verbose == True:
                print('Tiles are being converted to Tensorflow tensor...')

            self.tiles = tf.convert_to_tensor(self.tiles)
            
        # Save the data
        if save == True:
            
            if verbose == True:
                print('Tiles are being saved to local directory...')
                
            saver(self.tiles, file_type=data_type, file_name=f'{tile_style}_tiles') 
            
        if verbose == True:
                print("Tiling complete. Access tiles as a '.tiles' attribute.")
            
            
            
    def crop(self, image, tile_size):
        '''
        Crop the input image to the nearest multiple of tile_size along the spatial dimensions.

        Parameters:
            image (numpy.ndarray): The input image to be cropped.
            tile_size (int): The size of the tiles to crop to.

        Returns:
            numpy.ndarray: The cropped image.
        '''
        
        # Find the largest possible side size
        smaller_side = min(image.shape[-1],image.shape[-2])
        needed_size = np.floor(smaller_side/tile_size)*tile_size
        
        # Crop image
        crop = image[:, :-int(image.shape[-2] - needed_size),
                        :-int(image.shape[-1] - needed_size)]
        
        return crop
            

    def no_slide_tile(self, image, tile_size):
        '''
        Subset an image into non-overlapping tiles of specified size.

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
    
    
    def sliding_tile(self, image, tile_size):
        '''
        Subset an image into overlapping tiles of specified size with a stride of half the tile size.

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
    
    
    def sliding_transforms(self, image, tile_size):
        '''
        Subset an image into overlapping tiles with rotational and flip augmentations applied to each tile.

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

        return all_image_tiles, idx_tiles