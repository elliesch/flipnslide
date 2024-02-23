''' Tiling and Permutations -- Central Functionality '''

import numpy as np
from .tiling import ImageIngest

class Tiling:
    
    def __init__(self, tile_size:int=256,
                 tile_style:str = 'flipnslide',
                 save:bool = True,
                 **kwargs):
        
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
                
                coords = kwargs['coords']
                time_range = kwargs['time_range']
                
            else:
                raise ValueError("Either an 'image' or ('coords' and 'time_range') keyword arguments need to be provided.")
                
            ####!!!! for these i need to figure out if there's a way just to pass kwargs along!    
            if 'bands' in kwargs:
                bands = kwargs['bands']
                
            if 'cat_name' in kwargs:
                cat_name = kwargs['cat_name']
                
            if 'cloud_cov' in kwargs:
                cloud_cov = kwargs['cloud_cov']
                
            if 'res' in kwargs:
                res = kwargs['res']
                
            ####!!!! once i figure out how to pass the kwargs, i can pass the kwargs to here!
            image = ImageIngest(coords, time_range).image
            
        # Tiling method to be implemented
        
        if tile_style = 'flipnslide':
            
            self.tiles = sliding_transforms(image, self.tile_size)
            
        elif tile_style = 'overlap_tile':
            
            self.tiles = sliding_tile(image, self.tile_size)
        
        else:
            
            self.tiles = no_slide_tile(image, self.tile_size)
            

    def no_slide_tile(image, masks, tile_size):

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
    
    ### ADD SAVE CODES HERE
    ### np.save()
    ### torch.save(x, 'tensor.pt')
    ### torch.save(lst_tensors,'tensor_dataset.pt')
    ### tensor_list = torch.load('tensor_dataset.pt')