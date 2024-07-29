"""
Agglomerative Clustering implementation module.

This module provides an AgglomerativeClusterer class for performing 
Agglomerative Clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, AgglomerativeClustering, silhouette_score, MAX_CLUSTERS, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any

from src.config import (
    np, AgglomerativeClustering, silhouette_score, MAX_CLUSTERS,plt
)
from src.utils.scores import calculate_clustering_scores
class AgglomerativeClusterer:
    """
    A class for performing Agglomerative Clustering.

    This class provides a method to run Agglomerative Clustering on given data and find the optimal number of clusters.
    """

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Run Agglomerative Clustering on the given data.

        Parameters
        ----------
        _ : Any
            Unused parameter (kept for consistency with otherclusterers).
            
        features_scaled : np.ndarray
            The scaled feature array to cluster.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the clustering scores, labels, and the optimal number of clusters.
        """
        
        max_clusters = min(MAX_CLUSTERS, int(np.sqrt(len(features_scaled))))

        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agg_clustering.fit_predict(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, labels))

        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

        agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
        labels = agg_clustering.fit_predict(features_scaled)

        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'labels': labels,
            'optimal_k': optimal_k
        }