##############################################
# Top-Level Makefile for the llm repo
##############################################

all: default
default: install_requirements python_extensions test


############################################################################
# Environment Setup
############################################################################

.PHONY: upgrade_pip
upgrade_pip:
	pip install --upgrade pip


.PHONY: install_base_requirements
install_base_requirements: upgrade_pip
	pip install -r requirements.txt


.PHONY: install_dev_requirements
install_dev_requirements: upgrade_pip
	pip install -r requirements_dev.txt


.PHONY: install_requirements
install_requirements: install_base_requirements install_dev_requirements


############################################################################
# Build Package
############################################################################

.PHONY: python_extensions
python_extensions:
	python setup.py build_ext --inplace


############################################################################
# Testing
############################################################################

.PHONY: test
test:  python_extensions
	python -m unittest discover


.PHONY: benchmark
benchmark:
	python llm/tokenizers/benchmarks/benchmark_stdtoken.py
	python llm/tokenizers/benchmarks/benchmark_frequencies.py
	python llm/tokenizers/benchmarks/benchmark_merges.py


############################################################################
# Linting
############################################################################

.PHONY: check
check:
	black --check llm/
	black --check scripts/
	black --check serving/
	mypy llm/
	mypy scripts/
	mypy serving/
	flake8 llm/
	flake8 scripts/
	flake8 serving/


############################################################################
# Convenience Recipes
############################################################################

.PHONY: download_text
download_text:
	python scripts/download_and_split_data.py
