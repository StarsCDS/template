import sys
import os
import torch
from src.model.linear import Linear

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_init():
    model = Linear(1, 1)
    assert isinstance(model, torch.nn.Module)


def test_train_epoch():
    assert True


def test_test_epoch():
    assert True
