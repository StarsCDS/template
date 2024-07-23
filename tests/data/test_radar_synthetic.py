"""
Test module for the synthetic radar data generator and loader.

This module contains unit tests for the RadarDataset class and get_dataloader function
from the src.data.radar_synthetic module.

The tests ensure that the synthetic data generation and loading processes work as expected.
Imports:
    - sys, os: For modifying Python path
    - torch, pytest from src.config: For PyTorch operations and pytest framework
    - RadarDataset, get_dataloader from src.data.radar_synthetic: Classes and functions being tested
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import (
    torch,pytest
)
from src.data.radar_synthetic import RadarDataset, get_dataloader

def test_radar_dataset():
    """
    Test the RadarDataset class.

    This test ensures that:
    1. The dataset is not empty.
    2. Each item in the dataset is a torch.Tensor.
    3. Each item has the expected number of features (8).
    """
    dataset = RadarDataset()
    assert len(dataset) > 0
    assert isinstance(dataset[0], torch.Tensor)
    assert dataset[0].shape[0] == 8  # Assuming 8 features

def test_get_dataloader():
    """
    Test the get_dataloader function.

    This test ensures that:
    1. The function returns a DataLoader object.
    2. The DataLoader produces batches of the correct type and shape.
    """
    dataloader = get_dataloader()
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    
    batch = next(iter(dataloader))
    assert isinstance(batch, torch.Tensor)
    assert batch.dim() == 2
    assert batch.shape[1] == 8  # Assuming 8 features

def test_dataloader_batch_size():
    """
    Test the batch size of the DataLoader.

    This test ensures that the DataLoader respects the specified batch size.
    """
    batch_size = 32
    dataloader = get_dataloader(batch_size=batch_size)
    batch = next(iter(dataloader))
    assert batch.shape[0] == batch_size

def test_dataloader_shuffle():
    """
    Test the shuffle functionality of the DataLoader.

    This test ensures that when shuffle is set to True, 
    two separate DataLoaders produce different first batches.
    """
    dataloader1 = get_dataloader(shuffle=True)
    dataloader2 = get_dataloader(shuffle=True)
    
    batch1 = next(iter(dataloader1))
    batch2 = next(iter(dataloader2))
    
    assert not torch.all(torch.eq(batch1, batch2))