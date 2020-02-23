.PHONY: docs

install:
	python setup.py install

develop: install
	python setup.py develop
	pip install -r doc-requirements.txt
	pip install -r dev-requirements.txt

flake:
	flake8 setup.py --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	flake8 whatlies --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude __init__.py

test:
	pytest

check: flake test

docs:
	nbstripout notebooks/*.ipynb
	pytest --nbval --disable-warnings notebooks/*.ipynb
	jupyter nbconvert --to notebook notebooks/intro-with-tokens.ipynb --output ../docs/intro-with-tokens.ipynb
	jupyter nbconvert --to notebook notebooks/towards-embeddings.ipynb --output ../docs/towards-embeddings.ipynb
	mkdocs build --clean --site-dir public

serve-docs:
	mkdocs serve
