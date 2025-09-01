
# ===================================================
# File: Makefile
# ===================================================

.PHONY: help install clean test lint format docs

help:
	@echo "Available commands:"
	@echo "  make install    Install the package and dependencies"
	@echo "  make clean      Clean build artifacts and cache"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linting"
	@echo "  make format     Format code with black"
	@echo "  make docs       Build documentation"

install:
	conda env create -f environment.yml
	conda activate mtl-gnn-dta && pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/ -v --cov=mtl_gnn_dta --cov-report=html

lint:
	flake8 mtl_gnn_dta/ tests/
	mypy mtl_gnn_dta/

format:
	black mtl_gnn_dta/ tests/ examples/
	isort mtl_gnn_dta/ tests/ examples/

docs:
	cd docs && make clean && make html
