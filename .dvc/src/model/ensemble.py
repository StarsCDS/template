"""Ensemble Clustering implementation."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

from src.config import MAX_CLUSTERS
from src.utils.scores import calculate_clustering_scores

class EnsembleClusterer:
    """Ensemble Clustering class."""

    def __init__(self):
        self.max_clusters = MAX_CLUSTERS

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Run Ensemble Clustering algorithm."""
        results = self.evaluate_clustering_models(features_scaled)
        optimal_n = int(results.loc[results[['kmeans_silhouette', 'gmm_silhouette']].mean(axis=1).idxmax(), 'n_clusters'])

        ensemble_methods = [self.soft_voting_ensemble, self.majority_voting_ensemble, self.stacking_ensemble]
        ensemble_scores = []
        ensemble_labels = []

        for method in ensemble_methods:
            labels = method(features_scaled, optimal_n)
            score = silhouette_score(features_scaled, labels)
            ensemble_scores.append(score)
            ensemble_labels.append(labels)

        best_method_index = int(np.argmax(ensemble_scores))
        best_labels = ensemble_labels[best_method_index]

        scores = calculate_clustering_scores(features_scaled, best_labels)

        return {
            'scores': scores,
            'optimal_k': optimal_n,
            'ensemble_type': best_method_index + 1  # 1: Soft Voting, 2: Majority Voting, 3: Stacking
        }

    def evaluate_clustering_models(self, features_scaled: np.ndarray) -> pd.DataFrame:
        """Evaluate different clustering models."""
        results = []
        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(features_scaled)
            kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)

            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(features_scaled)
            gmm_labels = gmm.predict(features_scaled)
            gmm_silhouette = silhouette_score(features_scaled, gmm_labels)

            results.append((n_clusters, kmeans_silhouette, gmm_silhouette))

        return pd.DataFrame(results, columns=['n_clusters', 'kmeans_silhouette', 'gmm_silhouette'])

    def soft_voting_ensemble(self, features_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform soft voting ensemble."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(features_scaled)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(features_scaled)
        gmm_labels = gmm.predict(features_scaled)

        ensemble_labels = np.round((kmeans_labels + gmm_labels) / 2).astype(int)
        return ensemble_labels

    def majority_voting_ensemble(self, features_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform majority voting ensemble."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(features_scaled)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(features_scaled)
        gmm_labels = gmm.predict(features_scaled)

        gmm_labels_aligned = self.align_clusters(kmeans_labels, gmm_labels)
        ensemble_labels = np.where(kmeans_labels == gmm_labels_aligned, kmeans_labels, -1)

        return ensemble_labels

    def stacking_ensemble(self, features_scaled: np.ndarray, n_clusters: int, n_init: int = 10) -> np.ndarray:
        """Perform stacking ensemble."""
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        kmeans_distances = kmeans.transform(features_scaled)

        gmm = GaussianMixture(n_components=n_clusters, n_init=n_init, random_state=42)
        gmm.fit(features_scaled)
        gmm_proba = gmm.predict_proba(features_scaled)

        meta_features = np.hstack([kmeans_distances, gmm_proba])
        meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_clf.fit(meta_features, kmeans_labels)
        ensemble_labels = meta_clf.predict(meta_features)

        return ensemble_labels

    @staticmethod
    def align_clusters(kmeans_labels: np.ndarray, gmm_labels: np.ndarray) -> np.ndarray:
        """Align cluster labels from different algorithms."""
        size = max(kmeans_labels.max(), gmm_labels.max()) + 1
        matrix = np.zeros((size, size), dtype=np.int64)
        for k, g in zip(kmeans_labels, gmm_labels):
            matrix[k, g] += 1
        row_ind, col_ind = linear_sum_assignment(-matrix)
        aligned_labels = np.zeros_like(gmm_labels)
        for i, j in zip(row_ind, col_ind):
            aligned_labels[gmm_labels == j] = i
        return aligned_labels
