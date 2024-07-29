"""
OPTICS Clustering implementation module.

This module provides an OPTICSClusterer class for performing 
OPTICS (Ordering Points To Identify the Clustering Structure) clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, OPTICS, NearestNeighbors, MIN_SAMPLES_FACTOR, XI, MIN_CLUSTER_SIZE_FACTOR, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""
from typing import Dict, Any
from src.config import (
    np, OPTICS, NearestNeighbors,
    MIN_SAMPLES_FACTOR, XI, MIN_CLUSTER_SIZE_FACTOR,plt
)
from src.utils.scores import calculate_clustering_scores

class OPTICSClusterer:
    """
    A class for performing OPTICS clustering.

    This class provides a method to run OPTICS clustering on given data.
    """

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Run OPTICS Clustering algorithm.

        Parameters
        ----------
        _ : Any
            Unused parameter (kept for consistency with other clusterers).
        features_scaled : np.ndarray
            The scaled feature array to cluster.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the clustering scores, labels, cluster densities,
            and parameters used.
        """
        
        min_samples = max(5, int(MIN_SAMPLES_FACTOR * len(features_scaled)))
        min_cluster_size = max(5, int(MIN_CLUSTER_SIZE_FACTOR * len(features_scaled)))

        optics = OPTICS(min_samples=min_samples, xi=XI, min_cluster_size=min_cluster_size)
        labels = optics.fit_predict(features_scaled)

        neighbors = NearestNeighbors(n_neighbors=optics.min_samples).fit(features_scaled)
        core_distances_nn = np.sort(neighbors.kneighbors(features_scaled)[0][:, -1])

        scores = calculate_clustering_scores(features_scaled, labels)

        cluster_labels = np.unique(labels)
        cluster_densities = {}
        for cluster_label in cluster_labels:
            cluster_points = features_scaled[labels == cluster_label]
            if len(cluster_points) > 0:
                density = len(cluster_points) / (np.pi * (np.max(core_distances_nn[optics.ordering_][labels == cluster_label]) ** 2))
                cluster_densities[cluster_label] = density

        return {
            'scores': scores,
            'labels': labels,
            'cluster_densities': cluster_densities,
            'parameters': {
                'min_samples': min_samples,
                'xi': XI,
                'min_cluster_size': min_cluster_size
            }
        }