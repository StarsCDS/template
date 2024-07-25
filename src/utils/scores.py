from typing import Dict
from src.config import np, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def calculate_clustering_scores(features_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
    else:
        silhouette_avg = calinski_harabasz = davies_bouldin = -1
    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin
    }