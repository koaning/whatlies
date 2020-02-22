install:
	python setup.py install

develop: install
	python setup.py develop
	pip install -r doc-requirements.txt
	pip install -r dev-requirements.txt

flake:
	flake8 setup.py
	flake8 whatlies

test:
	pytest

check: flake test