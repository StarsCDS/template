"""
Test module for DBSCAN Clustering.

This module contains unit tests for the DBSCANClusterer class
from the src.model.dbscan module.

Imports:
    - sys, os: For modifying Python path
    - numpy (as np): For numerical operations
    - pytest: For test fixtures and assertions
    - BATCH_SIZE from src.config: For setting batch size in data loading
    - get_dataloader from src.data.radar_synthetic: For creating test data
    - DBSCANClusterer from src.model.dbscan: The class being tested
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import (
    np,pytest,BATCH_SIZE
)
from src.data.radar_synthetic import get_dataloader
from src.model.dbscan import DBSCANClusterer

@pytest.fixture
def dataloader():
    """
    Pytest fixture to create a DataLoader for test data.

    Returns:
        DataLoader: A DataLoader instance with synthetic radar data.
    """
    return get_dataloader(batch_size=BATCH_SIZE, shuffle=True)

@pytest.fixture
def features_scaled(dataloader):
    """
    Pytest fixture to prepare scaled feature data for clustering.

    Args:
        dataloader: The DataLoader fixture.

    Returns:
        np.ndarray: Concatenated and scaled feature data.
    """
    all_data = []
    for batch in dataloader:
        all_data.append(batch)
    all_data = np.concatenate(all_data, axis=0)
    return all_data

def test_dbscan_init():
    """
    Test the initialization of DBSCANClusterer.

    Ensures that the DBSCANClusterer instance has a 'k' attribute
    and that it's greater than 0.
    """
    dbscan = DBSCANClusterer()
    assert hasattr(dbscan, 'k')
    assert dbscan.k > 0

def test_dbscan_run(features_scaled):
    """
    Test the run method of DBSCANClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the run method returns a dictionary with expected keys and value types.
    """
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
    """
    Test the find_knee_point method of DBSCANClusterer.

    Ensures that the method correctly identifies the knee point in a simple array.
    """
    dbscan = DBSCANClusterer()
    distances = np.array([1, 2, 3, 4, 10, 11, 12])
    knee_point = dbscan.find_knee_point(distances)
    assert knee_point == 4