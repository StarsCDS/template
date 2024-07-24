import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.visualization import plot_dbscan, calculate_clustering_scores
class DBSCANClusterer:
    def __init__(self):
        self.k = 10

    def run(self, df, features_scaled, output_dir):
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(features_scaled)
        distances, indices = neigh.kneighbors(features_scaled)
        sorted_distances = np.sort(distances[:, self.k-1], axis=0)

        knee_point = self.find_knee_point(sorted_distances)
        eps = sorted_distances[knee_point]
        print(f"Optimal eps value based on the knee point: {eps:.4f}")
       
        while True:
            user_eps = input(f"Enter the eps value (default: {eps:.4f}): ")
            if not user_eps.strip():
                break
            try:
                user_eps = float(user_eps)
                if user_eps > 0:
                    eps = user_eps
                    break
                else:
                    print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

        while True:
            user_min_samples = input(f"Enter the min_samples value (default: {self.k}): ")
            if not user_min_samples.strip():
                min_samples = self.k
                break
            try:
                min_samples = int(user_min_samples)
                if min_samples > 0:
                    break
                else:
                    print("Please enter a positive integer")
            except ValueError:
                print("Please enter a valid integer")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(features_scaled)

        plot_dbscan(features_scaled, clusters, output_dir)
        scores = calculate_clustering_scores(features_scaled, clusters)

        return {
            'scores': scores,
            'eps': eps,
            'min_samples': min_samples
        }

    def find_knee_point(self, distances):
        diffs = np.diff(distances)
        knee_point = np.argmax(diffs) + 1
        return knee_point