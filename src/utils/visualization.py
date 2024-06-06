import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from typing import Dict

from config import np, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def plot_clusters_3d(features_scaled: np.ndarray, labels: np.ndarray, title: str):
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
    return plot_clusters_3d(features_scaled, labels, 'KMeans Clustering with 3D PCA')

def plot_gmm(features_scaled: np.ndarray, labels: np.ndarray):
    return plot_clusters_3d(features_scaled, labels, 'Gaussian Mixture Model Clustering with 3D PCA')

def plot_dbscan(features_scaled: np.ndarray, labels: np.ndarray):
    return plot_clusters_3d(features_scaled, labels, 'DBSCAN Clustering with 3D PCA')

def plot_ensemble(features_scaled: np.ndarray, labels: np.ndarray):
    return plot_clusters_3d(features_scaled, labels, 'Ensemble Clustering with 3D PCA')

def plot_agglomerative(features_scaled: np.ndarray, labels: np.ndarray):
    return plot_clusters_3d(features_scaled, labels, 'Agglomerative Clustering with 3D PCA')

def plot_optics(features_scaled: np.ndarray, labels: np.ndarray):
    return plot_clusters_3d(features_scaled, labels, 'OPTICS Clustering with 3D PCA')

def plot_hdbscan(features_scaled: np.ndarray, labels: np.ndarray):
    return plot_clusters_3d(features_scaled, labels, 'HDBSCAN Clustering with 3D PCA')