import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.visualization import plot_kmeans, calculate_clustering_scores

class KMeansClusterer:
    def __init__(self):
        self.max_clusters = 10

    def run(self, df, features_scaled, output_dir):
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
        while True:
            user_k = input(f"Enter the number of clusters (default: {optimal_k}): ")
            if not user_k.strip():
                break
            try:
                optimal_k = int(user_k)
                if 2 <= optimal_k <= self.max_clusters:
                    break
                else:
                    print(f"Please enter a number between 2 and {self.max_clusters}")
            except ValueError:
                print("Please enter a valid integer")

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        plot_kmeans(features_scaled, clusters, output_dir)
        
        scores = calculate_clustering_scores(features_scaled, clusters)
        return {'scores': scores, 'optimal_k': optimal_k}

    def find_elbow(self, k_values, inertias):
        diffs = np.diff(inertias)
        elbow_index = np.argmax(diffs) + 1
        return k_values[elbow_index]