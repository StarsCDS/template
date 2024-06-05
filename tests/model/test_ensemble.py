import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pytest
import numpy as np
import pandas as pd
from src.config import BATCH_SIZE
from src.data.radar_synthetic import get_dataloader
from src.model.ensemble import EnsembleClusterer

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

def test_ensemble_init():
    ensemble = EnsembleClusterer()
    assert hasattr(ensemble, 'max_clusters')
    assert ensemble.max_clusters > 0

def test_ensemble_run(features_scaled):
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
    ensemble = EnsembleClusterer()
    results = ensemble.evaluate_clustering_models(features_scaled)
    assert isinstance(results, pd.DataFrame)
    assert 'n_clusters' in results.columns
    assert 'kmeans_silhouette' in results.columns
    assert 'gmm_silhouette' in results.columns

def test_ensemble_soft_voting(features_scaled):
    ensemble = EnsembleClusterer()
    labels = ensemble.soft_voting_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)

def test_ensemble_majority_voting(features_scaled):
    ensemble = EnsembleClusterer()
    labels = ensemble.majority_voting_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)

def test_ensemble_stacking(features_scaled):
    ensemble = EnsembleClusterer()
    labels = ensemble.stacking_ensemble(features_scaled, n_clusters=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(features_scaled)