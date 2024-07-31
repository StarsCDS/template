"""
Configuration and import module for the clustering and deep learning project.

This module centralizes all imports and configurations used across the project.
It imports necessary libraries and defines constants used in various clustering 
algorithms and deep learning models.
"""

# Standard library imports
import sys
import os
import argparse
import random
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import h5py
import pytest

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Scipy imports
from scipy.optimize import linear_sum_assignment

# Other ML libraries
import hdbscan

# Deep learning model imports
from torchvision.models import resnet101, ResNet101_Weights
from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss

# Project-specific imports
from src.data.deepsig_data import RadioSignalDataset, load_and_split_data, print_class_distribution
from src.utils.visualisation.visualisation import get_embeddings, visualize_embeddings, visualize_embeddings_umap
from src.model.siamese import SiameseNetwork

# Constants
MAX_CLUSTERS = 10
MAX_COMPONENTS = 10
BATCH_SIZE = 32
RANDOM_STATE = 42
K_NEIGHBORS = 10

# HDBSCAN parameters
MIN_CLUSTER_SIZE_FACTOR = 0.02
MIN_SAMPLES_FACTOR = 0.01

# OPTICS parameters
XI = 0.05

# Make all imported modules and functions available
__all__ = [
    'np', 'pd', 'plt', 'Axes3D', 'torch', 'nn', 'F', 'optim', 'Dataset', 'DataLoader',
    'StepLR', 'torchvision', 'datasets', 'transforms', 'tqdm', 'h5py', 'pytest',
    'StandardScaler', 'PCA', 'KMeans', 'DBSCAN', 'AgglomerativeClustering', 'OPTICS',
    'GaussianMixture', 'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'RandomForestClassifier', 'NearestNeighbors', 'train_test_split',
    'linear_sum_assignment', 'hdbscan', 'resnet101', 'ResNet101_Weights',
    'TripletMarginLoss', 'TripletMarginWithDistanceLoss',
    'RadioSignalDataset', 'load_and_split_data', 'print_class_distribution',
    'get_embeddings', 'visualize_embeddings', 'visualize_embeddings_umap',
    'SiameseNetwork',
    'Any', 'Dict', 'List', 'Tuple', 'datetime', 'timedelta', 'argparse', 'random', 'sys', 'os',
    'MAX_CLUSTERS', 'MAX_COMPONENTS', 'BATCH_SIZE', 'RANDOM_STATE', 'K_NEIGHBORS',
    'MIN_CLUSTER_SIZE_FACTOR', 'MIN_SAMPLES_FACTOR', 'XI','h5py'
]