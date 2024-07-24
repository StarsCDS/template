
import hdbscan
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.visualization import plot_hdbscan, calculate_clustering_scores
class HDBSCANClusterer:
    def __init__(self):
        pass

    def run(self, df, features_scaled, output_dir):
        # Automatically determine parameters
        min_cluster_size = max(5, int(0.02 * len(features_scaled)))
        min_samples = max(5, int(0.01 * len(features_scaled)))

        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = hdbscan_clusterer.fit_predict(features_scaled)

        # Plot the results
        plot_hdbscan(features_scaled, labels, output_dir)

        # Calculate clustering scores
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'labels': labels,
            'parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples
            }
        }