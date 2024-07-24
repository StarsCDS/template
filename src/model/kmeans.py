"""KMeans clustering module."""
from typing import Dict, Any, List

from config import (
    np, KMeans, silhouette_score, MAX_CLUSTERS,
    plot_kmeans, calculate_clustering_scores
)

class KMeansClusterer:
    """KMeans clustering implementation."""

    def __init__(self):
        self.max_clusters = MAX_CLUSTERS

    def run(self, _, features_scaled: np.ndarray, output_dir: str) -> Dict[str, Any]:
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
        print(f"Optimal number of clusters based on silhouette score: {optimal_k_silhouette}")
        print(f"Optimal number of clusters based on elbow method: {optimal_k_elbow}")

        optimal_k = optimal_k_silhouette  # Use silhouette score's optimal k as default
        optimal_k = self.get_user_input(optimal_k)

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        plot_kmeans(features_scaled, clusters, output_dir)

        scores = calculate_clustering_scores(features_scaled, clusters)
        return {'scores': scores, 'optimal_k': optimal_k}

    def get_user_input(self, default_k: int) -> int:
        """Get user input for number of clusters."""
        while True:
            user_k = input(f"Enter the number of clusters (default: {default_k}): ")
            if not user_k.strip():
                return default_k
            try:
                k = int(user_k)
                if 2 <= k <= self.max_clusters:
                    return k
                print(f"Please enter a number between 2 and {self.max_clusters}")
            except ValueError:
                print("Please enter a valid integer")

    @staticmethod
    def find_elbow(k_values: List[int], inertias: List[float]) -> int:
        """Find the elbow point in the inertia curve."""
        diffs = np.diff(inertias)
        elbow_index = np.argmax(diffs) + 1
        return k_values[elbow_index]