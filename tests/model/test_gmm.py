"""
Test module for Gaussian Mixture Model Clustering.

This module contains unit tests for the GMMClusterer class
from the src.model.gmm module.

Imports:
    - sys, os: For modifying Python path
    - pytest: For test fixtures and assertions
    - numpy (as np): For numerical operations
    - BATCH_SIZE from src.config: For setting batch size in data loading
    - get_dataloader from src.data.radar_synthetic: For creating test data
    - GMMClusterer from src.model.gmm: The class being tested
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import (
    pytest,np,BATCH_SIZE
)
from src.data.radar_synthetic import get_dataloader
from src.model.gmm import GMMClusterer

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

def test_gmm_init():
    """
    Test the initialization of GMMClusterer.

    Ensures that the GMMClusterer instance has a 'max_components' attribute
    and that it's greater than 0.
    """
    gmm = GMMClusterer()
    assert hasattr(gmm, 'max_components')
    assert gmm.max_components > 0

def test_gmm_run(features_scaled):
    """
    Test the run method of GMMClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the run method returns a dictionary with expected keys and value types.
    """
    gmm = GMMClusterer()
    results = gmm.run(None, features_scaled)
    
    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']
    
    assert 'optimal_k' in results
    assert isinstance(results['optimal_k'], int)
    assert results['optimal_k'] > 0