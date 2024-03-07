''' Test dataset.py '''

import pytest
import torch
from ..dataset import TiledDataset


# Create sample data
tiles = [torch.randn(10, 10) for _ in range(100)]
permute_idx = torch.randint(0, 5, (100,))
batch_size = 10

def test_TiledDataset():
    
    # Test if TiledDataset initializes properly
    dataset = TiledDataset(tiles, batch_size=batch_size)
    
    # Test that the dataset exists for both training and testing
    assert len(dataset.train_dataset) > 0
    assert len(dataset.test_dataset) > 0
    
    # Test that the dataloader was created
    assert isinstance(dataset.dataloader, torch.utils.data.DataLoader)
    

if __name__ == '__main__':
    pytest.main()