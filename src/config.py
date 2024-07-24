"""Configuration and import file for the clustering project."""


# Standard library imports
import os
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
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
