.PHONY: docs

install:
	python -m pip install --upgrade pip
	python -m pip install -e .

develop: install
	python -m pip install -e ".[dev]"
	python -m pip install --upgrade pip wheel setuptools twine
	python setup.py develop
	pre-commit install

flake:
	flake8 setup.py --count --statistics --max-complexity=10 --max-line-length=127
	flake8 whatlies --count --statistics --max-complexity=10 --max-line-length=127 --exclude __init__.py
	flake8 tests --count --statistics --max-complexity=10 --max-line-length=127 --exclude __init__.py

test:
	pytest --nbval --nbval-lax --disable-warnings tests notebooks/*.ipynb

check: flake test

test-notebooks:
	pytest --nbval --nbval-lax --disable-warnings notebooks/*.ipynb

docs:
	mkdocs build --clean --site-dir public

serve-docs:
	mkdocs serve

pages: docs
	mkdocs gh-deploy --clean

clean:
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache
	nbstripout notebooks/*


pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*
