# Library Imports
import argparse
import random
from typing import Tuple
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from umap import UMAP
from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss
from sklearn.model_selection import train_test_split
from torchvision.models import resnet101, ResNet101_Weights

## Class Based imports
from src.data.deepsig_data import RadioSignalDataset, load_and_split_data, print_class_distribution

### Visualisation imports
from src.utils.visualisation.visualisation import get_embeddings, visualize_embeddings, visualize_embeddings_umap


###Model imports
from src.model.siamese import SiameseNetwork