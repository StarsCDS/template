import pytest
import torch
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.radar_synthetic import RadarDataset, get_dataloader

def test_radar_dataset():
    dataset = RadarDataset()
    assert len(dataset) > 0
    assert isinstance(dataset[0], torch.Tensor)
    assert dataset[0].shape[0] == 8  # Assuming 8 features

def test_get_dataloader():
    dataloader = get_dataloader()
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    
    batch = next(iter(dataloader))
    assert isinstance(batch, torch.Tensor)
    assert batch.dim() == 2
    assert batch.shape[1] == 8  # Assuming 8 features

def test_dataloader_batch_size():
    batch_size = 32
    dataloader = get_dataloader(batch_size=batch_size)
    batch = next(iter(dataloader))
    assert batch.shape[0] == batch_size

def test_dataloader_shuffle():
    dataloader1 = get_dataloader(shuffle=True)
    dataloader2 = get_dataloader(shuffle=True)
    
    batch1 = next(iter(dataloader1))
    batch2 = next(iter(dataloader2))
    
    assert not torch.all(torch.eq(batch1, batch2))