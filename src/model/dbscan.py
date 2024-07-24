"""DBSCAN Clustering implementation."""

from typing import Dict, Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from config import K_NEIGHBORS
from utils.visualization import plot_dbscan, calculate_clustering_scores

class DBSCANClusterer:
    """DBSCAN Clustering class."""

    def __init__(self):
        self.k = K_NEIGHBORS

    def run(self, df: Any, features_scaled: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """Run DBSCAN Clustering algorithm."""
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(features_scaled)
        distances, _ = neigh.kneighbors(features_scaled)
        sorted_distances = np.sort(distances[:, self.k-1], axis=0)

        knee_point = self.find_knee_point(sorted_distances)
        eps = sorted_distances[knee_point]
        print(f"Optimal eps value based on the knee point: {eps:.4f}")

        eps = self.get_user_input_float(f"Enter the eps value (default: {eps:.4f}): ", eps)
        min_samples = self.get_user_input_int(f"Enter the min_samples value (default: {self.k}): ", self.k)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(features_scaled)

        plot_dbscan(features_scaled, clusters, output_dir)
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

    @staticmethod
    def get_user_input_float(prompt: str, default: float) -> float:
        """Get float input from user with validation."""
        while True:
            user_input = input(prompt)
            if not user_input.strip():
                return default
            try:
                value = float(user_input)
                if value > 0:
                    return value
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

    @staticmethod
    def get_user_input_int(prompt: str, default: int) -> int:
        """Get integer input from user with validation."""
        while True:
            user_input = input(prompt)
            if not user_input.strip():
                return default
            try:
                value = int(user_input)
                if value > 0:
                    return value
                print("Please enter a positive integer")
            except ValueError:
                print("Please enter a valid integer")