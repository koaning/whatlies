.PHONY: docs

install:
	python -m pip install --upgrade pip
	python -m pip install -e .

develop:
	python -m pip install --upgrade pip wheel setuptools twine
	python -m pip install -e ".[dev]"
	pre-commit install
	python tests/scripts/prepare_fasttext_tests.py
	python tests/scripts/prepare_gensim_kv.py
	python tests/scripts/prepare_spacy_tests.py

download:
	python -m spacy download en_core_web_md
	python -m spacy download en_trf_robertabase_lg

flake:
	flake8 setup.py --count --statistics --max-complexity=10 --max-line-length=127
	flake8 whatlies --count --statistics --max-complexity=10 --max-line-length=127 --exclude __init__.py
	flake8 tests --count --statistics --max-complexity=10 --max-line-length=127 --exclude __init__.py

test:
	# pytest --nbval --nbval-lax --disable-warnings tests notebooks/*.ipynb
	pytest tests

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
	rm -rf build
	rm -rf dist

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*
