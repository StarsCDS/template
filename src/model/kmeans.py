"""
KMeans clustering implementation module.
- This module provides a KMeansClusterer class for performing KMeans clustering
  on given data.

Imports:
    - typing: For type hinting
    - NumPy, KMeans, silhouette_score, MAX_CLUSTERS and matplotlib plt from src.config
    - calculate_clustering_scores from src.utils.scores
"""

from typing import Dict, Any, List
from src.config import (
    np, KMeans, silhouette_score, MAX_CLUSTERS, plt
)
from src.utils.scores import calculate_clustering_scores
class KMeansClusterer:
    def __init__(self):
        self.max_clusters = MAX_CLUSTERS

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        inertia = []
        silhouette_scores = []
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

        optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        optimal_k_elbow = self.find_elbow(range(2, self.max_clusters + 1), inertia)

        optimal_k = optimal_k_silhouette  

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
    

        scores = calculate_clustering_scores(features_scaled, clusters)
        return {'scores': scores, 'optimal_k': optimal_k}

    @staticmethod
    def find_elbow(k_values: List[int], inertias: List[float]) -> int:
        diffs = np.diff(inertias)
        elbow_index = np.argmax(diffs) + 1
        return k_values[elbow_index]
    
    