''' PyTorch Dataset and DataLoader tools '''

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader



class TiledDataset:
    
    # def __init__(self, tiles,
    #              permute_idx,
    #              batch_size,
    #              set_type:str='full',
    #              split:float=0.1,
    #              randomize_permute:bool=False,
    #              state:int=18,
    #              **kwargs):
    
    def __init__(self, tiles,
                 batch_size,
                 set_type:str='full',
                 split:float=0.1,
                 # randomize_permute:bool=False,
                 state:int=18,
                 **kwargs):
        '''
        Initialize a TiledDataset object.

        Parameters:
        - tiles (list): A list containing the dataset tiles.
        - set_type (str): Type of dataset. Default is 'full'.
        - split (float): Percentage of data to use for testing. Default is 0.1.
        - state (int): Random state for reproducibility. Default is 18.
        - batch_size: Size of batches to use in data loading.

        **kwargs: Additional keyword arguments.
        - permute_split (list): list of permutation number indices.

        Raises:
        - AssertionError: If randomize_permute is True.

        Notes:
        - This method initializes a TiledDataset object, which represents a dataset split into tiles.
        - It splits the data into training and testing sets, constructs PyTorch datasets, and builds dataloaders.
        - Note that DataLoaders preserving knowledge of permutation number are not available in this release.
        ''' 
        
        #First split the data
        if 'permute_split' in kwargs:
            train_tiles, test_tiles = self.train_test(tiles, split, state, **kwargs)
        
        else:
            train_tiles, test_tiles = self.train_test(tiles, split, state)
        
        #Next add data to PyTorch Datasets
        self.train_dataset = TileDataset(train_tiles)
        self.test_dataset = TileDataset(test_tiles)
        
        #Finally build into PyTorch Dataloaders
        self.dataloader = DataLoader(self.train_dataset, batch_size=batch_size)              
        
        
    def train_test(self, tiles, 
                   split, state,
                   randomize_permute=False,
                   **kwargs):
        '''
        Split data into training and testing sets.

        Parameters:
        - tiles (list): A list containing the dataset tiles.
        - split (float): Percentage of data to use for testing.
        - state (int): Random state for reproducibility.
        
        **kwargs: Additional keyword arguments.
        - permute_split (list): list of permutation number indices.

        Returns:
        - train_arrays (list): List of arrays containing training data.
        - test_arrays (list): List of arrays containing testing data.

        Notes:
        - This function splits the input data into training and testing sets.
        - If randomize_permute is True, the function splits data based on unique permutation indices.
        - Otherwise, it splits the data directly.
        '''
        
        #Add warning that randomize permute is not functional yet
        assert not (randomize_permute
                   ), "DataLoaders that preserve knowledge of permutation number aren't available in this release."
    
        train_arrays = []
        test_arrays = []

        if randomize_permute == True:
            permute_idx = kwargs['permute_split']
            permutation = len(np.unique(permute_idx))
            
            train_data_arrays = []
            test_data_arrays = []

            for ii in range(permutation):
                perm_mask = permute_idx == ii
                X_train, X_test = train_test_split(tiles[perm_mask], 
                                                   test_size=split, random_state=state,
                                                   shuffle=True)
                train_data_arrays.append(X_train)
                test_data_arrays.append(X_test)
                
                return train_data_arrays, test_data_arrays

        else:
            X_train, X_test = train_test_split(tiles, test_size=split, 
                                               random_state=state,
                                               shuffle=True)
            train_arrays = X_train
            test_arrays = X_test


            return train_arrays, test_arrays
    

    
class TileDataset(Dataset):
    '''
    Dataset class for creating iterable custom PyTorch datasets for streaming through GPU.

    Args:
    - tiles (list): List containing the dataset tiles.
    - transform (callable, optional): Optional transform to be applied to the tiles.

    Attributes:
    - tiles (list): List containing the dataset tiles.
    - transform (callable or None): Transform function to be applied to the tiles.

    Methods:
    - __len__(self): Returns the length of the dataset.
    - __getitem__(self, index): Returns the item at the specified index.

    Note:
    - This class follows the PyTorch Dataset protocol as documented.
    '''
    
    def __init__(self, tiles, transform=None):
        self.tiles = tiles
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]

        if self.transform:
            tile = self.transform(tile)

        return tile
        