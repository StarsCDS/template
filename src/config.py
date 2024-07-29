"""
Configuration and import module for the clustering project.

This module centralizes all imports and configurations used across the project.
It imports necessary libraries and defines constants used in various clustering algorithms.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import hdbscan
from datetime import datetime, timedelta  
import pytest
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
    'np', 'pd', 'plt', 'Axes3D', 'torch', 'Dataset', 'DataLoader', 'StandardScaler', 'PCA',
    'KMeans', 'DBSCAN', 'AgglomerativeClustering', 'OPTICS', 'GaussianMixture',
    'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'RandomForestClassifier', 'NearestNeighbors', 'linear_sum_assignment', 'hdbscan',
    'Any', 'Dict', 'List', 'Tuple', 'datetime', 'timedelta',
    'MAX_CLUSTERS', 'MAX_COMPONENTS', 'BATCH_SIZE', 'RANDOM_STATE', 'K_NEIGHBORS',
    'MIN_CLUSTER_SIZE_FACTOR', 'MIN_SAMPLES_FACTOR', 'XI'
]