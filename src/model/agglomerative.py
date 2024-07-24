import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.visualization import plot_agglomerative, calculate_clustering_scores
class AgglomerativeClusterer:
    def __init__(self):
        pass

    def run(self, df, features_scaled, output_dir):
        max_clusters = min(20, int(np.sqrt(len(features_scaled))))
        
        # Find optimal number of clusters
        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agg_clustering.fit_predict(features_scaled)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
        
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

        # Apply Agglomerative Clustering with the optimal number of clusters
        agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
        labels = agg_clustering.fit_predict(features_scaled)

        # Plot the results
        plot_agglomerative(features_scaled, labels, output_dir)

        # Calculate clustering scores
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'labels': labels,
            'optimal_k': optimal_k
        }