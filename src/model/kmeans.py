"""KMeans clustering module."""
# from typing import Dict, Any, List
# import matplotlib.pyplot as plt

# from config import (
#     np, KMeans, silhouette_score, MAX_CLUSTERS
# )
# from utils.scores import calculate_clustering_scores
"""KMeans clustering module."""
from typing import Dict, Any, List
import matplotlib.pyplot as plt

from src.config import (
    np, KMeans, silhouette_score, MAX_CLUSTERS
)
from src.utils.scores import calculate_clustering_scores

# ... rest of the file remains the same ...
class KMeansClusterer:
    """KMeans clustering implementation."""

    def __init__(self):
        self.max_clusters = MAX_CLUSTERS

    def run(self, _, features_scaled: np.ndarray) -> Dict[str, Any]:
        """Run KMeans clustering algorithm."""
        inertia = []
        silhouette_scores = []
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

        optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        optimal_k_elbow = self.find_elbow(range(2, self.max_clusters + 1), inertia)

        optimal_k = optimal_k_silhouette  # Use silhouette score's optimal k as default

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # fig = plot_kmeans(features_scaled, clusters)
        # plt.close(fig)  # Close the figure to free up memory

        scores = calculate_clustering_scores(features_scaled, clusters)
        return {'scores': scores, 'optimal_k': optimal_k}

    @staticmethod
    def find_elbow(k_values: List[int], inertias: List[float]) -> int:
        """Find the elbow point in the inertia curve."""
        diffs = np.diff(inertias)
        elbow_index = np.argmax(diffs) + 1
        return k_values[elbow_index]
    
    