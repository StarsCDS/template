import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.visualization import plot_optics, calculate_clustering_scores

class OPTICSClusterer:
    def __init__(self):
        pass

    def run(self, df, features_scaled, output_dir):
        # Automatically determine parameters
        min_samples = max(5, int(0.1 * len(features_scaled)))
        xi = 0.05
        min_cluster_size = max(5, int(0.05 * len(features_scaled)))

        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        labels = optics.fit_predict(features_scaled)

        # Calculate density and shape metrics
        core_distances = optics.reachability_[optics.ordering_]
        neighbors = NearestNeighbors(n_neighbors=optics.min_samples).fit(features_scaled)
        core_distances_nn = np.sort(neighbors.kneighbors(features_scaled)[0][:, -1])

        # Plot the results
        plot_optics(features_scaled, labels, output_dir)

        # Calculate clustering scores
        scores = calculate_clustering_scores(features_scaled, labels)

        # Calculate and print cluster densities
        cluster_labels = np.unique(labels)
        cluster_densities = {}
        for cluster_label in cluster_labels:
            cluster_points = features_scaled[labels == cluster_label]
            if len(cluster_points) > 0:
                density = len(cluster_points) / (np.pi * (np.max(core_distances_nn[optics.ordering_][labels == cluster_label]) ** 2))
                cluster_densities[cluster_label] = density

        return {
            'scores': scores,
            'labels': labels,
            'cluster_densities': cluster_densities,
            'parameters': {
                'min_samples': min_samples,
                'xi': xi,
                'min_cluster_size': min_cluster_size
            }
        }