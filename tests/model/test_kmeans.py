import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pytest
import numpy as np
from src.config import BATCH_SIZE
from src.data.radar_synthetic import get_dataloader
from src.model.kmeans import KMeansClusterer
# from config import BATCH_SIZE
# from data.radar_synthetic import get_dataloader
# from model.kmeans import KMeansClusterer
@pytest.fixture
def dataloader():
    return get_dataloader(batch_size=BATCH_SIZE, shuffle=True)

@pytest.fixture
def features_scaled(dataloader):
    all_data = []
    for batch in dataloader:
        all_data.append(batch)
    all_data = np.concatenate(all_data, axis=0)
    return all_data

def test_kmeans_init():
    kmeans = KMeansClusterer()
    assert hasattr(kmeans, 'max_clusters')
    assert kmeans.max_clusters > 0

def test_kmeans_run(features_scaled):
    kmeans = KMeansClusterer()
    results = kmeans.run(None, features_scaled)
    
    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']
    
    assert 'optimal_k' in results
    assert isinstance(results['optimal_k'], int)
    assert results['optimal_k'] > 0

def test_kmeans_find_elbow():
    kmeans = KMeansClusterer()
    k_values = range(2, 11)
    inertias = [100, 80, 60, 50, 45, 42, 40, 39, 38]
    elbow = kmeans.find_elbow(k_values, inertias)
    assert elbow in k_values
