"""
DBSCAN Clustering implementation module.

This module provides a DBSCANClusterer class for performing 
DBSCAN Clustering on given data.

Imports:
    - typing: For type hinting
    - NumPy, NearestNeighbors, DBSCAN, K_NEIGHBORS, and plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""

from typing import Dict, Any
from src.config import (
    np, NearestNeighbors, DBSCAN, K_NEIGHBORS, plt
)
from src.utils.scores import calculate_clustering_scores
class DBSCANClusterer:
    """
    A class for performing DBSCAN Clustering.

    This class provides methods to run DBSCAN Clustering on given data
    and find the optimal epsilon value.

    Attributes
    ----------
    k : int
        The number of neighbors to consider, imported from src.config.
    """

    def __init__(self):
        """Initialize the DBSCANClusterer."""
        self.k = K_NEIGHBORS

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Run DBSCAN Clustering on the given data.

        Parameters
        ----------
        _ : Any
            Unused parameter (kept for consistency with other clusterers).
        features_scaled : np.ndarray
            The scaled feature array to cluster.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the clustering scores, epsilon value,
            and minimum samples.
        """
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(features_scaled)
        distances, _ = neigh.kneighbors(features_scaled)
        sorted_distances = np.sort(distances[:, self.k-1], axis=0)

        knee_point = self.find_knee_point(sorted_distances)
        eps = sorted_distances[knee_point]
        min_samples = self.k

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(features_scaled)

        scores = calculate_clustering_scores(features_scaled, clusters)

        return {
            'scores': scores,
            'eps': eps,
            'min_samples': min_samples
        }

    @staticmethod
    def find_knee_point(distances: np.ndarray) -> int:
        """
        Find the knee point in the k-distance graph.

        Parameters
        ----------
        distances : np.ndarray
            The sorted k-distances for each point.

        Returns
        -------
        int
            The index of the knee point in the k-distance graph.
        """
        diffs = np.diff(distances)
        knee_point = np.argmax(diffs) + 1
        return knee_point