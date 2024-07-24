"""Agglomerative Clustering implementation."""

from typing import Dict, Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from config import MAX_CLUSTERS
from utils.visualization import plot_agglomerative, calculate_clustering_scores

class AgglomerativeClusterer:
    """Agglomerative Clustering class."""

    def run(self, df: Any, features_scaled: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """Run Agglomerative Clustering algorithm."""
        max_clusters = min(MAX_CLUSTERS, int(np.sqrt(len(features_scaled))))
        
        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agg_clustering.fit_predict(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
        
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

        agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
        labels = agg_clustering.fit_predict(features_scaled)

        plot_agglomerative(features_scaled, labels, output_dir)
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'labels': labels,
            'optimal_k': optimal_k
        }