import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def evaluate_clustering_models(X_scaled, max_clusters=10):
    results = []
    for n_clusters in range(2, max_clusters + 1):
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

        # GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_silhouette = silhouette_score(X_scaled, gmm_labels)

        results.append((n_clusters, kmeans_silhouette, gmm_silhouette))

    return pd.DataFrame(results, columns=['n_clusters', 'kmeans_silhouette', 'gmm_silhouette'])

def soft_voting_ensemble(X_scaled, n_clusters):
    # K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)

    # Ensemble by averaging cluster assignments
    ensemble_labels = np.round((kmeans_labels + gmm_labels) / 2).astype(int)

    # Recalculate labels to ensure they are sequential starting from 0
    unique_labels = np.unique(ensemble_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    final_labels = np.array([label_mapping[label] for label in ensemble_labels])

    return final_labels

def majority_voting_ensemble(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)

    # Align the GMM labels with KMeans labels
    gmm_labels_aligned = align_clusters(kmeans_labels, gmm_labels)

    # Majority Voting
    ensemble_labels = np.where(kmeans_labels == gmm_labels_aligned, kmeans_labels, -1)

    return ensemble_labels

def stacking_ensemble(X_scaled, n_clusters, n_init=10):
    # K-means Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_distances = kmeans.transform(X_scaled)

    # GMM Clustering
    gmm = GaussianMixture(n_components=n_clusters, n_init=n_init, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_proba = gmm.predict_proba(X_scaled)

    # Combine features for meta-classifier
    meta_features = np.hstack([kmeans_distances, gmm_proba])

    # Train meta-classifier
    meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_clf.fit(meta_features, kmeans_labels)  # Using KMeans labels as "ground truth"

    # Final predictions
    ensemble_labels = meta_clf.predict(meta_features)

    return ensemble_labels

def align_clusters(kmeans_labels, gmm_labels):
    size = max(kmeans_labels.max(), gmm_labels.max()) + 1
    matrix = np.zeros((size, size), dtype=np.int64)
    for k, g in zip(kmeans_labels, gmm_labels):
        matrix[k, g] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    aligned_labels = np.zeros_like(gmm_labels)
    for i, j in zip(row_ind, col_ind):
        aligned_labels[gmm_labels == j] = i
    return aligned_labels

def run_ensemble(data, features_scaled, **kwargs):
    # Determine optimal number of clusters
    results = evaluate_clustering_models(features_scaled)
    optimal_n = results.loc[results[['kmeans_silhouette', 'gmm_silhouette']].mean(axis=1).idxmax(), 'n_clusters']
    print(f"Optimal number of clusters determined: {optimal_n}")

    # Ask user for input
    n_clusters = int(input(f"Enter the number of clusters (default: {optimal_n}): ") or optimal_n)
    ensemble_type = int(input("Choose the type of ensembling:\n1. Soft Voting (Average)\n2. Majority Voting\n3. Stacking\n"))

    # Apply the selected ensemble model
    if ensemble_type == 1:
        ensemble_labels = soft_voting_ensemble(features_scaled, n_clusters)
    elif ensemble_type == 2:
        ensemble_labels = majority_voting_ensemble(features_scaled, n_clusters)
    else:
        ensemble_labels = stacking_ensemble(features_scaled, n_clusters)

    # Visualize the results in 3D
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=ensemble_labels, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.colorbar(scatter, label='Cluster label')
    plt.title('Ensemble Clustering Results with 3D PCA')
    plt.show()

    # Evaluate and return the performance scores
    if len(set(ensemble_labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, ensemble_labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, ensemble_labels)
        davies_bouldin = davies_bouldin_score(features_scaled, ensemble_labels)
    else:
        silhouette_avg = -1
        calinski_harabasz = -1
        davies_bouldin = -1

    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    }