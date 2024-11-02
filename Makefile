all: default
default: install_requirements


.PHONY: upgrade_pip
upgrade_pip:
	pip install --upgrade pip


.PHONY: install_base_requirements
install_base_requirements: upgrade_pip
	pip install -r requirements.txt


.PHONY: install_dev_requirements
install_dev_requirements: upgrade_pip
	pip install -r requirements_dev.txt


.PHONY: python_extensions
python_extensions:
	python setup.py build_ext --inplace


.PHONY: test
test:  python_extensions
	PYTHONPATH=. python -m unittest discover

