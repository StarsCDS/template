import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from src.data.fmnist import FMNIST


def test_init():
    dataset = FMNIST()
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_len():
    assert True


def test_getitem():
    assert True
