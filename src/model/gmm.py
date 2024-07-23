"""
Gaussian Mixture Model (GMM) Clustering implementation module.

This module provides a GMMClusterer class for performing 
Gaussian Mixture Model clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, GaussianMixture, silhouette_score, MAX_COMPONENTS, RANDOM_STATE, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any

from src.config import (
    np, GaussianMixture, silhouette_score,
    MAX_COMPONENTS, RANDOM_STATE,plt
)
from src.utils.scores import calculate_clustering_scores
class GMMClusterer:
    """
    A class for performing Gaussian Mixture Model (GMM) clustering.

    This class provides methods to run GMM clustering on given data
    and find the optimal number of components.

    Attributes
    ----------
    max_components : int
        The maximum number of components to consider, imported from src.config.
    """

    def __init__(self):
        """Initialize the GMMClusterer."""
        self.max_components = MAX_COMPONENTS

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Run GMM clustering on the given data.

        Parameters
        ----------
        _ : Any Unused parameter (kept for consistency with other clusterers).
        
        features_scaled : np.ndarray
            The scaled feature array to cluster.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the clustering scores and the optimal
            number of components.
        """
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
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'optimal_k': optimal_k
        }