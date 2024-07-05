import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

def run_kmeans(data, features_scaled, **kwargs):
    # Determine the optimal number of clusters using elbow and silhouette methods
    max_clusters = 10
    inertia = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

    # Plot elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve')
    plt.show()

    # Plot silhouette curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Curve')
    plt.show()

    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    # Ask the user to enter the number of clusters or use the default optimal k
    user_k = input(f"Enter the number of clusters (default: {optimal_k}): ")
    if user_k.strip():
        optimal_k = int(user_k)

    # Apply KMeans with the specified number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # Calculate clustering performance scores
    silhouette_avg = silhouette_score(features_scaled, clusters)
    calinski_harabasz = calinski_harabasz_score(features_scaled, clusters)
    davies_bouldin = davies_bouldin_score(features_scaled, clusters)

    # Visualize the results using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    principal_df = pd.DataFrame(principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    final_df = pd.concat([principal_df, pd.DataFrame({'Cluster': clusters})], axis=1)

    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown']
    for k in range(optimal_k):
        cluster_data = final_df[final_df['Cluster'] == k]
        plt.scatter(cluster_data['Principal Component 1'], cluster_data['Principal Component 2'], s=50, c=colors[k], label=f'Cluster {k}')
    plt.title('KMeans Clustering with 2D PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    }