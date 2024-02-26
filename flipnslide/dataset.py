''' PyTorch Dataset and DataLoader tools '''

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class TiledDataset:
    
    def __init__(self, tiles,
                 permute_idx,
                 set_type:str='full',
                 split:float=0.1,
                 randomize_permute:bool=False,
                 state:int=18,
                 batch_size,
                 **kwargs):
        '''
        '''
        #Add warning that randomize permute is not functional yet
        assert not (randomize_permute
                   ), "DataLoaders that preserve knowledge of permutation number aren't available in this release."
            
        
        #First split the data
        train_tiles, test_tiles = train_test(tiles, permute_idx,
                                             split, randomize_permute, state)
        
        #Next add data to PyTorch Datasets
        self.train_dataset = TileDataset(train_tiles)
        self.test_dataset = TileDataset(test_tiles)
        
        #Finally build into PyTorch Dataloaders
        self.dataloader = DataLoader(self.train_dataset, batch_size=batch_size)        
        
        
        
        
    def train_test(self, tiles, permute_idx, 
                   split, randomize_permute, state):
    
        train_arrays = []
        test_arrays = []

        if randomize_permute == True:
            permutation = len(np.unique(permute_idx))

            for ii in range(permutation):
                perm_mask = permute_idx == ii
                X_train, X_test = train_test_split(tiles[perm_mask], 
                                                   test_size=split, random_state=state,
                                                   shuffle=True)
                train_data_arrays.append(X_train)
                test_data_arrays.append(X_test)

        else:
            X_train, X_test = train_test_split(tiles, test_size=split, 
                                               random_state=state,
                                               shuffle=True)
            train_arrays = X_train
            test_arrays = X_test


        return train_arrays, test_arrays
    
    
class TileDataset(Dataset):
    '''
    Class to create iterable custom PyTorch Dataset for streaming through GPU.
    Follows PyTorch protocol from documentation there.
    '''
    
    def __init__(self, tiles, transform=None):
        self.tiles = channel_data
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]

        if self.transform:
            tile = self.transform(tile)

        return tile
        