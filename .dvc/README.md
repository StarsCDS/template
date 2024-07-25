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

- Download the raw datasets (optional)
  ```sh
  dvc pull
  ```

## Current Tasks

- [ ] Add tests for custom datasets and dataloaders
- [ ] Create supervised model for fashion MNIST dataset using pytorch-lightning
- [ ] Use dvc to save the models and version them

## Tools

- [git](https://git-scm.com/doc) for code version control
- [dvc](https://dvc.org/doc/start) for data version control
- [poetry](https://python-poetry.org/) for python dependency management
- [pylint](https://pypi.org/project/pylint/), [black](https://github.com/psf/black) for code quality
- [pytest](https://docs.pytest.org/en/8.2.x/) for all types of tests
- [typer](https://typer.tiangolo.com/) for easy cli
- [clearml](https://clear.ml/) for experiment tracking and model monitoring

## Contributing Instructions

- Fork this repository and make sure that the fork is in-sync before sending a pull request.
- Note all the steps that you think are worth noting along with any good tutorials for the
  particular task that you are doing.
- Make sure that you are following [coding best practices for python](https://www.datacamp.com/blog/python-best-practices-for-better-code).

## For Documentation

- [Pytest routine](https://github.com/StarsCDS/Documentations.git) for extended documentation in pytest routine
