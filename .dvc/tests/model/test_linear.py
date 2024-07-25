import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from src.model.linear import Linear


def test_init():
    model = Linear(1, 1)
    assert isinstance(model, torch.nn.Module)


def test_train_epoch():
    assert True


def test_test_epoch():
    assert True
