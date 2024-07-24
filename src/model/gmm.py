"""Gaussian Mixture Model Clustering implementation."""

from typing import Dict, Any

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from config import MAX_COMPONENTS, RANDOM_STATE
from utils.visualization import plot_gmm, calculate_clustering_scores

class GMMClusterer:
    """Gaussian Mixture Model Clustering class."""

    def __init__(self):
        self.max_components = MAX_COMPONENTS

    def run(self, df: Any, features_scaled: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """Run Gaussian Mixture Model Clustering algorithm."""
        silhouette_scores = []
        bic_scores = []

        for k in range(2, self.max_components + 1):
            gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
            gmm.fit(features_scaled)
            labels = gmm.predict(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
            bic_scores.append(gmm.bic(features_scaled))

        optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        optimal_k_bic = bic_scores.index(min(bic_scores)) + 2

        print(f"Optimal number of components based on silhouette score: {optimal_k_silhouette}")
        print(f"Optimal number of components based on BIC score: {optimal_k_bic}")

        optimal_k = optimal_k_silhouette  # Use silhouette score's optimal k as default
        optimal_k = self.get_user_input(optimal_k)

        gmm = GaussianMixture(n_components=optimal_k, random_state=RANDOM_STATE)
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)

        plot_gmm(features_scaled, labels, output_dir)
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'optimal_k': optimal_k
        }

    def get_user_input(self, default: int) -> int:
        """Get user input for number of components."""
        while True:
            user_k = input(f"Enter the number of components (default: {default}): ")
            if not user_k.strip():
                return default
            try:
                n_components = int(user_k)
                if 2 <= n_components <= self.max_components:
                    return n_components
                print(f"Please enter a number between 2 and {self.max_components}")
            except ValueError:
                print("Please enter a valid integer")