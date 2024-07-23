import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.siamese import SiameseNetwork
import pytest
import torch

@pytest.fixture
def model():
    return SiameseNetwork(num_classes=10,embedding_dim=128)


@pytest.fixture
def anchor():
    return torch.randn(1,1024,2)


@pytest.fixture
def positive():
    return torch.randn(1,1024,2)


@pytest.fixture
def negative():
    return torch.randn(1,1024,2)


@pytest.fixture
def labels():
    return torch.tensor([1])