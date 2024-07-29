"""
HDBSCAN Clustering implementation module.

This module provides an HDBSCANClusterer class for performing 
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) 
clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, hdbscan, MIN_CLUSTER_SIZE_FACTOR, MIN_SAMPLES_FACTOR, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""

from typing import Dict, Any
from src.config import (
    np, hdbscan, MIN_CLUSTER_SIZE_FACTOR, MIN_SAMPLES_FACTOR,plt
)
from src.utils.scores import calculate_clustering_scores
class HDBSCANClusterer:
    """
    A class for performing HDBSCAN clustering.

    This class provides a method to run HDBSCAN clustering on given data.
    """

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Run HDBSCAN Clustering algorithm.

        Parameters
        ----------
        _ : Any
            Unused parameter (kept for consistency with other clusterers).
        features_scaled : np.ndarray
            The scaled feature array to cluster.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the clustering scores, labels, and parameters used.
        """
        min_cluster_size = max(5, int(MIN_CLUSTER_SIZE_FACTOR * len(features_scaled)))
        min_samples = max(5, int(MIN_SAMPLES_FACTOR * len(features_scaled)))

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(features_scaled)

        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'labels': labels,
            'parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples
            }
        }