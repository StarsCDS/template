import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.visualization import plot_ensemble, calculate_clustering_scores
class EnsembleClusterer:
    def __init__(self):
        self.max_clusters = 10

    def run(self, df, features_scaled, output_dir):
        results = self.evaluate_clustering_models(features_scaled)
        optimal_n = results.loc[results[['kmeans_silhouette', 'gmm_silhouette']].mean(axis=1).idxmax(), 'n_clusters']
        print(f"Optimal number of clusters determined: {optimal_n}")

        while True:
            user_k = input(f"Enter the number of clusters (default: {optimal_n}): ")
            if not user_k.strip():
                n_clusters = optimal_n
                break
            try:
                n_clusters = int(user_k)
                if 2 <= n_clusters <= self.max_clusters:
                    break
                else:
                    print(f"Please enter a number between 2 and {self.max_clusters}")
            except ValueError:
                print("Please enter a valid integer")

        ensemble_types = ["Soft Voting", "Majority Voting", "Stacking"]
        print("\nChoose the ensemble type:")
        for i, etype in enumerate(ensemble_types, 1):
            print(f"{i}. {etype}")

        while True:
            ensemble_choice = input("Enter the number corresponding to your choice: ")
            try:
                ensemble_choice = int(ensemble_choice)
                if 1 <= ensemble_choice <= 3:
                    break
                else:
                    print("Please enter a number between 1 and 3")
            except ValueError:
                print("Please enter a valid integer")

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

    def evaluate_clustering_models(self, X_scaled):
        results = []
        for n_clusters in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm_labels = gmm.fit_predict(X_scaled)
            gmm_silhouette = silhouette_score(X_scaled, gmm_labels)

            results.append((n_clusters, kmeans_silhouette, gmm_silhouette))

        return pd.DataFrame(results, columns=['n_clusters', 'kmeans_silhouette', 'gmm_silhouette'])

    def soft_voting_ensemble(self, X_scaled, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)

        ensemble_labels = np.round((kmeans_labels + gmm_labels) / 2).astype(int)

        unique_labels = np.unique(ensemble_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        final_labels = np.array([label_mapping[label] for label in ensemble_labels])

        return final_labels

    def majority_voting_ensemble(self, X_scaled, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)

        gmm_labels_aligned = self.align_clusters(kmeans_labels, gmm_labels)

        ensemble_labels = np.where(kmeans_labels == gmm_labels_aligned, kmeans_labels, -1)

        return ensemble_labels

    def stacking_ensemble(self, X_scaled, n_clusters, n_init=10):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_distances = kmeans.transform(X_scaled)

        gmm = GaussianMixture(n_components=n_clusters, n_init=n_init, random_state=42)
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_proba = gmm.predict_proba(X_scaled)

        meta_features = np.hstack([kmeans_distances, gmm_proba])

        meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_clf.fit(meta_features, kmeans_labels)

        ensemble_labels = meta_clf.predict(meta_features)

        return ensemble_labels

    def align_clusters(self, kmeans_labels, gmm_labels):
        size = max(kmeans_labels.max(), gmm_labels.max()) + 1
        matrix = np.zeros((size, size), dtype=np.int64)
        for k, g in zip(kmeans_labels, gmm_labels):
            matrix[k, g] += 1
        row_ind, col_ind = linear_sum_assignment(-matrix)
        aligned_labels = np.zeros_like(gmm_labels)
        for i, j in zip(row_ind, col_ind):
            aligned_labels[gmm_labels == j] = i
        return aligned_labels