<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 6b4dd4b (Update README.md)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

<<<<<<< HEAD
=======
>>>>>>> 96f0b24 (add template for custom dataset and model)
=======
>>>>>>> 6b4dd4b (Update README.md)
import torch
from src.data.fmnist import FMNIST


def test_init():
    dataset = FMNIST()
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_len():
    assert True


def test_getitem():
    assert True
