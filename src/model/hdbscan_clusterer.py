from typing import Dict, Any
import matplotlib.pyplot as plt

from src.config import (
    np, hdbscan, MIN_CLUSTER_SIZE_FACTOR, MIN_SAMPLES_FACTOR
)
from src.utils.scores import calculate_clustering_scores
class HDBSCANClusterer:

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Run HDBSCAN Clustering algorithm."""
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