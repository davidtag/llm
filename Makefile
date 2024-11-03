all: default
default: install_requirements

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
	PYTHONPATH=. python -m unittest discover


.PHONY: download_text
download_text:
	PYTHONPATH=. python scripts/download_and_split_data.py


.PHONY: check
check:
	black --check llm/
	black --check scripts/
	mypy llm/
	mypy scripts/
