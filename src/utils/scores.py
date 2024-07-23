"""
Clustering Score Calculation Module.

This module provides functionality to calculate various clustering performance scores.

Functions:
    calculate_clustering_scores: Calculates multiple clustering performance scores.
"""
from typing import Dict
from src.config import ( 
    np, silhouette_score, calinski_harabasz_score, davies_bouldin_score 
)

def calculate_clustering_scores(features_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate various clustering performance scores.

    This function computes the Silhouette Score, Calinski-Harabasz Index,
    and Davies-Bouldin Index for a given clustering result.

    Args:
        features_scaled (np.ndarray): The scaled feature array used for clustering.
        labels (np.ndarray): The cluster labels assigned to each data point.

    Returns:
        Dict[str, float]: A dictionary containing the computed scores.
    """
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