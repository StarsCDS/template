import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pytest
import numpy as np
from src.config import BATCH_SIZE
from src.data.radar_synthetic import get_dataloader
from src.model.hdbscan_clusterer import HDBSCANClusterer

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

def test_hdbscan_init():
    hdbscan = HDBSCANClusterer()
    assert hasattr(hdbscan, 'run')

def test_hdbscan_run(features_scaled):
    hdbscan = HDBSCANClusterer()
    results = hdbscan.run(None, features_scaled)
    
    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']
    
    assert 'labels' in results
    assert isinstance(results['labels'], np.ndarray)
    assert len(results['labels']) == len(features_scaled)
    
    assert 'parameters' in results
    assert isinstance(results['parameters'], dict)
    assert 'min_cluster_size' in results['parameters']
    assert 'min_samples' in results['parameters']