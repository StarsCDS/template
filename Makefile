lint:
	poetry run pylint --fail-under=7.0 --recursive=y .

test:
	poetry run pytest ./tests
	PYTHONPATH=src poetry run pytest ./tests/data
