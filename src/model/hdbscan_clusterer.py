"""HDBSCAN Clustering implementation."""

from typing import Dict, Any
import numpy as np

from config import (
    hdbscan, MIN_CLUSTER_SIZE_FACTOR, MIN_SAMPLES_FACTOR,
    plot_hdbscan, calculate_clustering_scores
)

class HDBSCANClusterer:
    """HDBSCAN Clustering class."""

    def run(self, _, features_scaled: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """Run HDBSCAN Clustering algorithm."""
        min_cluster_size = max(5, int(MIN_CLUSTER_SIZE_FACTOR * len(features_scaled)))
        min_samples = max(5, int(MIN_SAMPLES_FACTOR * len(features_scaled)))

        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = hdbscan_clusterer.fit_predict(features_scaled)

        plot_hdbscan(features_scaled, labels, output_dir)
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'labels': labels,
            'parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples
            }
        }