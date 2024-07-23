"""
Test module for Ensemble Clustering.

This module contains unit tests for the EnsembleClusterer class
from the src.model.ensemble module.

Imports:
    - sys, os: For modifying Python path
    - numpy (as np): For numerical operations
    - pandas (as pd): For DataFrame operations
    - pytest: For test fixtures and assertions
    - BATCH_SIZE from src.config: For setting batch size in data loading
    - get_dataloader from src.data.radar_synthetic: For creating test data
    - EnsembleClusterer from src.model.ensemble: The class being tested
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import (
    np, pd, pytest, BATCH_SIZE
)
from src.data.radar_synthetic import get_dataloader
from src.model.ensemble import EnsembleClusterer

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

def test_ensemble_init():
    """
    Test the initialization of EnsembleClusterer.

    Ensures that the EnsembleClusterer instance has a 'max_clusters' attribute
    and that it's greater than 0.
    """
    ensemble = EnsembleClusterer()
    assert hasattr(ensemble, 'max_clusters')
    assert ensemble.max_clusters > 0

def test_ensemble_run(features_scaled):
    """
    Test the run method of EnsembleClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the run method returns a dictionary with expected keys and value types.
    Also prints detailed information about the results for debugging purposes.
    """
    ensemble = EnsembleClusterer()
    results = ensemble.run(None, features_scaled)
    
    print("Results:", results)
    print("Type of optimal_k:", type(results['optimal_k']))
    print("Is optimal_k an int?", isinstance(results['optimal_k'], int))
    print("Is optimal_k a numpy int?", isinstance(results['optimal_k'], np.integer))
    print("Value of optimal_k:", results['optimal_k'])
    print("Type of ensemble_type:", type(results['ensemble_type']))
    print("Is ensemble_type an int?", isinstance(results['ensemble_type'], int))
    print("Value of ensemble_type:", results['ensemble_type'])
    
    assert 'scores' in results
    assert isinstance(results['scores'], dict)
    assert 'Silhouette Score' in results['scores']
    assert 'Calinski-Harabasz Index' in results['scores']
    assert 'Davies-Bouldin Index' in results['scores']
    
    assert 'optimal_k' in results
    assert results['optimal_k'] == int(results['optimal_k']), "optimal_k is not an integer-like value"
    assert isinstance(results['optimal_k'], (int, np.integer)), "optimal_k is not an int or numpy integer"
    assert results['optimal_k'] > 0
    
    assert 'ensemble_type' in results
    assert isinstance(results['ensemble_type'], int), "ensemble_type is not a Python int"
    assert 1 <= results['ensemble_type'] <= 3

def test_ensemble_evaluate_clustering_models(features_scaled):
    """
    Test the evaluate_clustering_models method of EnsembleClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the method returns a DataFrame with expected columns.
    """
    ensemble = EnsembleClusterer()
    results = ensemble.evaluate_clustering_models(features_scaled)
    assert isinstance(results, pd.DataFrame)
    assert 'n_clusters' in results.columns
    assert 'kmeans_silhouette' in results.columns
    assert 'gmm_silhouette' in results.columns

def test_ensemble_soft_voting(features_scaled):
    """
    Test the soft_voting_ensemble method of EnsembleClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the method returns a numpy array of labels with the correct length.
    """
    ensemble = EnsembleClusterer()
    labels = ensemble.soft_voting_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)

def test_ensemble_majority_voting(features_scaled):
    """
    Test the majority_voting_ensemble method of EnsembleClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the method returns a numpy array of labels with the correct length.
    """
    ensemble = EnsembleClusterer()
    labels = ensemble.majority_voting_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)

def test_ensemble_stacking(features_scaled):
    """
    Test the stacking_ensemble method of EnsembleClusterer.

    Args:
        features_scaled: The features_scaled fixture.

    Ensures that the method returns a numpy array of labels with the correct length.
    """
    ensemble = EnsembleClusterer()
    labels = ensemble.stacking_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)