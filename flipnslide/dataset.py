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
                 **kwargs):
        '''
        '''
        
        ...
        
        
    def train_test_arrays(tiles, permute_idx, 
                          split, randomize_permute=False):
    
        train_arrays = []
        test_arrays = []

        if randomize_permute == True:
            permutation = len(np.unique(permute_idx))

            for ii in range(permutation):
                perm_mask = permute_idx == ii
                X_train, X_test = train_test_split(channels[perm_mask], 
                                                   test_size=split, random_state=state,
                                                   shuffle=True)
                train_data_arrays.append(X_train)
                test_data_arrays.append(X_test)

        else:
            X_train, X_test = train_test_split(channels[ii], 
                                               test_size=split, random_state=state,
                                               shuffle=True)
            train_arrays = X_train
            test_arrays = X_test


        return train_arrays, test_arrays
    
    
    def make_dataset(self, tiles):
        
        '''
        '''
        
        ...
        