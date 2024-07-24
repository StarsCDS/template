import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def plot_clusters_3d(features_scaled, labels, title, filename, output_dir):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                         c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    
    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.colorbar(scatter, label='Cluster label')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def calculate_clustering_scores(features_scaled, labels):
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
    else:
        silhouette_avg = calinski_harabasz = davies_bouldin = -1
    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    }

# Specific plotting functions for each clustering method
def plot_kmeans(features_scaled, labels, output_dir):
    plot_clusters_3d(features_scaled, labels, 'KMeans Clustering with 3D PCA', 'kmeans_clustering_3d.png', output_dir)

def plot_gmm(features_scaled, labels, output_dir):
    plot_clusters_3d(features_scaled, labels, 'Gaussian Mixture Model Clustering with 3D PCA', 'gmm_clustering_3d.png', output_dir)

def plot_dbscan(features_scaled, labels, output_dir):
    plot_clusters_3d(features_scaled, labels, 'DBSCAN Clustering with 3D PCA', 'dbscan_clustering_3d.png', output_dir)

def plot_ensemble(features_scaled, labels, output_dir):
    plot_clusters_3d(features_scaled, labels, 'Ensemble Clustering with 3D PCA', 'ensemble_clustering_3d.png', output_dir)
    
def plot_agglomerative(features_scaled, labels, output_dir):
    plot_clusters_3d(features_scaled, labels, 'Agglomerative Clustering with 3D PCA', 'agglomerative_clustering_3d.png', output_dir)

def plot_optics(features_scaled, labels, output_dir):
    plot_clusters_3d(features_scaled, labels, 'OPTICS Clustering with 3D PCA', 'optics_clustering_3d.png', output_dir)

def plot_hdbscan(features_scaled, labels, output_dir):
    plot_clusters_3d(features_scaled, labels, 'HDBSCAN Clustering with 3D PCA', 'hdbscan_clustering_3d.png', output_dir)