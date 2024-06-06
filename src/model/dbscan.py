from typing import Dict, Any
import matplotlib.pyplot as plt

from src.config import (
    np, NearestNeighbors, DBSCAN, K_NEIGHBORS
)
from src.utils.scores import calculate_clustering_scores
class DBSCANClusterer:

    def __init__(self):
        self.k = K_NEIGHBORS

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Run DBSCAN Clustering algorithm."""
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
        """Find the knee point in the k-distance graph."""
        diffs = np.diff(distances)
        knee_point = np.argmax(diffs) + 1
        return knee_point