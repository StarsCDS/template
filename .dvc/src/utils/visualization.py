"""Visualization utilities for clustering results."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from typing import Dict

from config import np, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str):
    """Plot 3D scatter plot of clustering results."""
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
        c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50
    )
    
    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.colorbar(scatter, label='Cluster label')
    
    return fig

def plot_kmeans(features_scaled: np.ndarray, labels: np.ndarray):
    """Plot KMeans clustering results."""
    return plot_clusters_3d(features_scaled, labels, 'KMeans Clustering with 3D PCA')

def plot_gmm(features_scaled: np.ndarray, labels: np.ndarray):
    """Plot Gaussian Mixture Model clustering results."""
    return plot_clusters_3d(features_scaled, labels, 'Gaussian Mixture Model Clustering with 3D PCA')

def plot_dbscan(features_scaled: np.ndarray, labels: np.ndarray):
    """Plot DBSCAN clustering results."""
    return plot_clusters_3d(features_scaled, labels, 'DBSCAN Clustering with 3D PCA')

def plot_ensemble(features_scaled: np.ndarray, labels: np.ndarray):
    """Plot Ensemble clustering results."""
    return plot_clusters_3d(features_scaled, labels, 'Ensemble Clustering with 3D PCA')

def plot_agglomerative(features_scaled: np.ndarray, labels: np.ndarray):
    """Plot Agglomerative clustering results."""
    return plot_clusters_3d(features_scaled, labels, 'Agglomerative Clustering with 3D PCA')

def plot_optics(features_scaled: np.ndarray, labels: np.ndarray):
    """Plot OPTICS clustering results."""
    return plot_clusters_3d(features_scaled, labels, 'OPTICS Clustering with 3D PCA')

def plot_hdbscan(features_scaled: np.ndarray, labels: np.ndarray):
    """Plot HDBSCAN clustering results."""
    return plot_clusters_3d(features_scaled, labels, 'HDBSCAN Clustering with 3D PCA')