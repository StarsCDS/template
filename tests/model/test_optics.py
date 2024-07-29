"""
Test module for OPTICS Clustering.

This module contains unit tests for the OPTICSClusterer class
from the src.model.optics module.

Imports:
    - sys, os: For modifying Python path
    - numpy (as np): For numerical operations
    - pytest: For test fixtures and assertions
    - BATCH_SIZE from src.config: For setting batch size in data loading
    - get_dataloader from src.data.radar_synthetic: For creating test data
    - OPTICSClusterer from src.model.optics: The class being tested
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import (
    np,pytest,BATCH_SIZE
)
from src.data.radar_synthetic import get_dataloader
from src.model.optics import OPTICSClusterer

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

def test_optics_init():
    """
    Test the initialization of OPTICSClusterer.

    Ensures that the OPTICSClusterer instance has a 'run' method.
    """
    optics = OPTICSClusterer()
    assert hasattr(optics, 'run')

def test_optics_run(features_scaled):
    """
    Test the run method of OPTICSClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the run method returns a dictionary with expected keys and value types.
    """
    optics = OPTICSClusterer()
    results = optics.run(None, features_scaled)
    
    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']
    
    assert 'labels' in results
    assert isinstance(results['labels'], np.ndarray)
    assert len(results['labels']) == len(features_scaled)
    
    assert 'cluster_densities' in results
    assert isinstance(results['cluster_densities'], dict)
    
    assert 'parameters' in results
    assert isinstance(results['parameters'], dict)
    assert 'min_samples' in results['parameters']
    assert 'xi' in results['parameters']
    assert 'min_cluster_size' in results['parameters']