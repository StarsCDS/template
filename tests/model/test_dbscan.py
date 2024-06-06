import pytest
import numpy as np
from src.config import BATCH_SIZE
from src.data.radar_synthetic import get_dataloader
from src.model.dbscan import DBSCANClusterer

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

def test_dbscan_init():
    dbscan = DBSCANClusterer()
    assert hasattr(dbscan, 'k')
    assert dbscan.k > 0

def test_dbscan_run(features_scaled):
    dbscan = DBSCANClusterer()
    results = dbscan.run(None, features_scaled)
    
    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']
    
    assert 'eps' in results
    assert isinstance(results['eps'], float)
    assert results['eps'] > 0
    
    assert 'min_samples' in results
    assert isinstance(results['min_samples'], int)
    assert results['min_samples'] > 0

def test_dbscan_find_knee_point():
    dbscan = DBSCANClusterer()
    distances = np.array([1, 2, 3, 4, 10, 11, 12])
    knee_point = dbscan.find_knee_point(distances)
    assert knee_point == 4