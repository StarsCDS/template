"""
Clustering Visualization Module.

This module provides functions to visualize clustering results using 3D plots.

Functions:
    plot_clusters_3d: Creates a 3D scatter plot of clustered data.
    plot_kmeans: Plots KMeans clustering results.
    plot_gmm: Plots Gaussian Mixture Model clustering results.
    plot_dbscan: Plots DBSCAN clustering results.
    plot_ensemble: Plots Ensemble clustering results.
    plot_agglomerative: Plots Agglomerative clustering results.
    plot_optics: Plots OPTICS clustering results.
    plot_hdbscan: Plots HDBSCAN clustering results.
"""
from typing import Dict
from src.config import (
    np, plt, Axes3D, PCA,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str):
    """
    Create a 3D scatter plot of clustered data.

    Args:
        features_scaled (np.ndarray): The scaled feature array used for clustering.
        labels (np.ndarray): The cluster labels assigned to each data point.
        title (str): The title of the plot.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
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