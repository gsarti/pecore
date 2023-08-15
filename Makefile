.PHONY: quality style

# Check that source code meets quality standards

quality:
	black --diff --check --config pyproject.toml ./
	ruff  --no-fix --config pyproject.toml ./

# Format source code automatically

style:
	black --config pyproject.toml ./
	ruff --config pyproject.toml ./

# Setup the library

setup:
	pip install requirements.txt
	pip install -e .