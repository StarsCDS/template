"""Gaussian Mixture Model Clustering implementation."""

from typing import Dict, Any
import matplotlib.pyplot as plt

from src.config import (
    np, GaussianMixture, silhouette_score,
    MAX_COMPONENTS, RANDOM_STATE
)
from src.utils.scores import calculate_clustering_scores
class GMMClusterer:
    """Gaussian Mixture Model Clustering class."""

    def __init__(self):
        self.max_components = MAX_COMPONENTS

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
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

        optimal_k = optimal_k_silhouette  # Use silhouette score's optimal k as default

        gmm = GaussianMixture(n_components=optimal_k, random_state=RANDOM_STATE)
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)

        # fig = plot_gmm(features_scaled, labels)
        # plt.close(fig)  # Close the figure to free up memory

        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'optimal_k': optimal_k
        }