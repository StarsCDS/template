import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_dbscan(data, features_scaled, **kwargs):
    # Generate k-distance graph
    k = 10  # Adjust based on your preference
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features_scaled)
    distances, indices = neigh.kneighbors(features_scaled)
    sorted_distances = np.sort(distances[:, k-1], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.title(f'k-Distance Graph (k={k})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {k}-th Nearest Neighbor')
    plt.grid(True)
    plt.show()

    # Determine optimal eps and min_samples
    optimal_eps = float(input("Enter the optimal eps value based on the k-distance graph: "))
    optimal_min_samples = int(input("Enter the optimal min_samples value: "))

    # Ask the user to enter eps and min_samples or use the default values
    user_eps = input(f"Enter the eps value (default: {optimal_eps}): ")
    user_min_samples = input(f"Enter the min_samples value (default: {optimal_min_samples}): ")

    if user_eps.strip():
        eps = float(user_eps)
    else:
        eps = optimal_eps

    if user_min_samples.strip():
        min_samples = int(user_min_samples)
    else:
        min_samples = optimal_min_samples

    # Apply DBSCAN with the specified eps and min_samples
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features_scaled)

    # Visualize the results using PCA
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(features_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2],
                         c=clusters, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    ax.set_title('DBSCAN Clustering Results with PCA-reduced Data (3D)')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.colorbar(scatter, label='Cluster label')
    plt.show()

    # Calculate clustering performance scores
    if len(set(clusters)) > 1:
        silhouette_avg = silhouette_score(features_scaled, clusters)
        calinski_harabasz = calinski_harabasz_score(features_scaled, clusters)
        davies_bouldin = davies_bouldin_score(features_scaled, clusters)
    else:
        silhouette_avg = -1
        calinski_harabasz = -1
        davies_bouldin = -1

    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    }