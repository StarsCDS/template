"""Ensemble Clustering implementation."""

from typing import Dict, Any

from config import (
    np, pd, KMeans, GaussianMixture, RandomForestClassifier,
    silhouette_score, linear_sum_assignment, MAX_CLUSTERS,
    plot_ensemble, calculate_clustering_scores
)

class EnsembleClusterer:
    """Ensemble Clustering class."""

    def __init__(self):
        self.max_clusters = MAX_CLUSTERS

    def run(self, df: Any, features_scaled: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """Run Ensemble Clustering algorithm."""
        results = self.evaluate_clustering_models(features_scaled)
        optimal_n = results.loc[results[['kmeans_silhouette', 'gmm_silhouette']].mean(axis=1).idxmax(), 'n_clusters']
        print(f"Optimal number of clusters determined: {optimal_n}")

        n_clusters = self.get_user_input_int(f"Enter the number of clusters (default: {optimal_n}): ", optimal_n)

        ensemble_types = ["Soft Voting", "Majority Voting", "Stacking"]
        print("\nChoose the ensemble type:")
        for i, etype in enumerate(ensemble_types, 1):
            print(f"{i}. {etype}")

        ensemble_choice = self.get_user_input_int("Enter the number corresponding to your choice: ", 1, 3)

        if ensemble_choice == 1:
            labels = self.soft_voting_ensemble(features_scaled, n_clusters)
        elif ensemble_choice == 2:
            labels = self.majority_voting_ensemble(features_scaled, n_clusters)
        else:
            labels = self.stacking_ensemble(features_scaled, n_clusters)

        print(f"\nUsing {ensemble_types[ensemble_choice-1]} Ensemble")
        
        plot_ensemble(features_scaled, labels, output_dir)
        scores = calculate_clustering_scores(features_scaled, labels)

        return {
            'scores': scores,
            'optimal_k': n_clusters,
            'ensemble_type': ensemble_choice
        }

    def evaluate_clustering_models(self, features_scaled: np.ndarray) -> pd.DataFrame:
        """Evaluate different clustering models."""
        results = []
        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(features_scaled)
            kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)

            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm_labels = gmm.fit_predict(features_scaled)
            gmm_silhouette = silhouette_score(features_scaled, gmm_labels)

            results.append((n_clusters, kmeans_silhouette, gmm_silhouette))

        return pd.DataFrame(results, columns=['n_clusters', 'kmeans_silhouette', 'gmm_silhouette'])

    def soft_voting_ensemble(self, features_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform soft voting ensemble."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(features_scaled)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(features_scaled)

        ensemble_labels = np.round((kmeans_labels + gmm_labels) / 2).astype(int)

        unique_labels = np.unique(ensemble_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        final_labels = np.array([label_mapping[label] for label in ensemble_labels])

        return final_labels

    def majority_voting_ensemble(self, features_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform majority voting ensemble."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(features_scaled)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(features_scaled)

        gmm_labels_aligned = self.align_clusters(kmeans_labels, gmm_labels)

        ensemble_labels = np.where(kmeans_labels == gmm_labels_aligned, kmeans_labels, -1)

        return ensemble_labels

    def stacking_ensemble(self, features_scaled: np.ndarray, n_clusters: int, n_init: int = 10) -> np.ndarray:
        """Perform stacking ensemble."""
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        kmeans_distances = kmeans.transform(features_scaled)

        gmm = GaussianMixture(n_components=n_clusters, n_init=n_init, random_state=42)
        gmm_proba = gmm.fit_predict_proba(features_scaled)

        meta_features = np.hstack([kmeans_distances, gmm_proba])

        meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_clf.fit(meta_features, kmeans_labels)

        ensemble_labels = meta_clf.predict(meta_features)

        return ensemble_labels

    @staticmethod
    def align_clusters(kmeans_labels: np.ndarray, gmm_labels: np.ndarray) -> np.ndarray:
        """Align cluster labels from different algorithms."""
        size = max(kmeans_labels.max(), gmm_labels.max()) + 1
        matrix = np.zeros((size, size), dtype=np.int64)
        for k, g in zip(kmeans_labels, gmm_labels):
            matrix[k, g] += 1
        row_ind, col_ind = linear_sum_assignment(-matrix)
        aligned_labels = np.zeros_like(gmm_labels)
        for i, j in zip(row_ind, col_ind):
            aligned_labels[gmm_labels == j] = i
        return aligned_labels

    @staticmethod
    def get_user_input_int(prompt: str, default: int, max_value: int = None) -> int:
        """Get integer input from user with validation."""
        while True:
            user_input = input(prompt)
            if not user_input.strip():
                return default
            try:
                value = int(user_input)
                if max_value is not None:
                    if 1 <= value <= max_value:
                        return value
                    print(f"Please enter a number between 1 and {max_value}")
                elif value > 0:
                    return value
                else:
                    print("Please enter a positive integer")
            except ValueError:
                print("Please enter a valid integer")