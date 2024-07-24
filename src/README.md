# Clustering Project

## Project Structure

- `main.py`: Entry point of the application
- `config.py`: Contains all dependencies and configurations
- `src/`: Main source code directory
  - `data/`: Data handling
  - `model/`: Clustering models
  - `utils/`: Utility functions

## Features

- Synthetic radar_synthetic dataloader in `data/`
- Clustering models implemented:
  1. KMeans
  2. DBSCAN
  3. GMM (Gaussian Mixture Model)
  4. Ensemble
  5. Agglomerative
  6. OPTICS
  7. HDBSCAN
- Visualization utilities in `utils/visualization.py`
- Results utilities in `utils/scores.py`

## Usage

Run `main.py` to start the clustering process. Select a model when prompted.

## Dependencies

See `config.py` for a list of required libraries.
