import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D

def run_gmm(data, features_scaled, **kwargs):
    max_components = 10
    silhouette_scores = []
    bic_scores = []

    for k in range(2, max_components + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)
        silhouette_scores.append(silhouette_score(features_scaled, labels))
        bic_scores.append(gmm.bic(features_scaled))

    optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
    optimal_k_bic = bic_scores.index(min(bic_scores)) + 2

    # Plot silhouette curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_components + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Curve')
    plt.show()

    # Plot BIC curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_components + 1), bic_scores, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.title('BIC Curve')
    plt.show()

    print(f"Optimal number of components based on silhouette score: {optimal_k_silhouette}")
    print(f"Optimal number of components based on BIC score: {optimal_k_bic}")

    # Ask the user to enter the number of components or use the default optimal k from silhouette curve
    user_k = input(f"Enter the number of components (default: {optimal_k_silhouette}): ")
    if user_k.strip():
        n_components = int(user_k)
    else:
        n_components = optimal_k_silhouette

    # Apply Gaussian Mixture Model with the specified number of components
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features_scaled)
    labels = gmm.predict(features_scaled)

    # Perform PCA for dimensionality reduction to 3D for visualization
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)

    # Scatter plot of the three PCA components
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    plt.title('Gaussian Mixture Model Clustering Results with PCA-reduced Data')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.colorbar(scatter, label='Cluster label')
    plt.show()

    # Calculate clustering performance scores
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
    else:
        silhouette_avg = -1
        calinski_harabasz = -1
        davies_bouldin = -1

    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    }

