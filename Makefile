.PHONY: docs

install:
	python setup.py install

develop: install
	python setup.py develop
	pip install -r doc-requirements.txt
	pip install -r dev-requirements.txt

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

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*