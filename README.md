# Template repository for ML Projects

## Setup

- Install [`poetry`](https://python-poetry.org/docs/#installing-with-the-official-installer)

- Install the virtual environment
  ```sh
  poetry install
  ```

- Activate the virtual environment
  ```sh
  source $(poetry env info --path)/bin/activate
  ```

- Activate `pre-commit` hooks
  ```sh
  pre-commit install
  ```

## Current Tasks

- [ ] Add functions to create custom pytorch datasets and dataloaders from raw data
- [ ] Add tests for these custom datasets and dataloaders
- [ ] Add CI pipelines for checking code linting and tests
- [ ] Create supervised model for fashion MNIST dataset using pytorch
- [ ] Create supervised model for fashion MNIST dataset using pytorch-lightning
- [ ] Use dvc to save the models and version them

## Tools

- dvc for data version control
- git for code version control
- clearml for experiment tracking and model monitoring
