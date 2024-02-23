''' Visualization Tools '''

import matplotlib.pyplot as plt
import numpy as np

#Think of visualization flag like verbose flag
#Maybe even add verbose flag to the tiling class too?

def ingest_viz(image):
    '''
    Display all of the channels of the downloaded image.

    Parameters:
        image (numpy.ndarray): The input image with shape (num_channels, height, width).

    Returns:
        None
    '''
    
    n_channels = image.shape[0]
    
    fig, axes = plt.subplots(nrows=1, ncols=n_channels, 
                             figsize=(int(3*n_channels), 3)) 
    
    for ii in range(n_channels):
        axes[ii].imshow(image[ii])
        axes[ii].axis('off')
        
    plt.show()


def crop_viz(image, crop):
    '''
    Display the original and cropped images side by side without axis ticks and labels.

    Parameters:
        image (numpy.ndarray): The original image to be displayed on the left.
        crop (numpy.ndarray): The cropped image to be displayed on the right.

    Returns:
        None
    '''
    
    fig, axes = plt.subplots(nrows=1, ncols=2, 
                             figsize=(7, 3)) 
    
    
    axes[0].imshow(image[0])
    axes[0].axis('off')
    axes[0].set_title('Uncropped Image')
    
    axes[1].imshow(crop[0])
    axes[1].axis('off')
    axes[1].set_title('Cropped Image')
        
    plt.show()


def tile_viz(tiles):
    '''
    Display six random tiles from the tiled image.

    Parameters:
        tiles (numpy.ndarray): An array of tiles with shape [num_tiles, channels, height, width].

    Returns:
        None
    '''
    
    random_idx = np.random.choice(tiles.shape[0], 6, replace=False)
    
    fig, axes = plt.subplots(nrows=1, ncols=6, 
                             figsize=(18, 3)) 
    
    for ii in range(6):
        axes[ii].imshow(tiles[random_idx[ii],0,...])
        axes[ii].axis('off')
        
    plt.show()