.PHONY: check fix

# Check that source code meets quality standards

check:
	ruff format --check --config pyproject.toml ./
	ruff  --no-fix --config pyproject.toml ./

# Format source code automatically

fix:
	ruff format --config pyproject.toml ./
	ruff check --config pyproject.toml ./

# Setup the library

setup:
	pip install requirements.txt
	pip install -e .