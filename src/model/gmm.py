import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.visualization import plot_gmm, calculate_clustering_scores
class GMMClusterer:
    def __init__(self):
        self.max_components = 10

    def run(self, df, features_scaled, output_dir):
        silhouette_scores = []
        bic_scores = []

        for k in range(2, self.max_components + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(features_scaled)
            labels = gmm.predict(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
            bic_scores.append(gmm.bic(features_scaled))

        optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        optimal_k_bic = bic_scores.index(min(bic_scores)) + 2

        print(f"Optimal number of components based on silhouette score: {optimal_k_silhouette}")
        print(f"Optimal number of components based on BIC score: {optimal_k_bic}")

        optimal_k = optimal_k_silhouette  # Use silhouette score's optimal k as default
        while True:
            user_k = input(f"Enter the number of components (default: {optimal_k}): ")
            if not user_k.strip():
                break
            try:
                n_components = int(user_k)
                if 2 <= n_components <= self.max_components:
                    optimal_k = n_components
                    break
                else:
                    print(f"Please enter a number between 2 and {self.max_components}")
            except ValueError:
                print("Please enter a valid integer")

        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        gmm.fit(features_scaled)
        labels = gmm.predict(features_scaled)

        plot_gmm(features_scaled, labels, output_dir)
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'optimal_k': optimal_k
        }