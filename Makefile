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
	pytest --nbval --disable-warnings tests notebooks/*.ipynb

check: flake test

docs:
	nbstripout notebooks/*.ipynb
	pytest --nbval --disable-warnings notebooks/*.ipynb
	jupyter nbconvert --to notebook --execute notebooks/intro-with-tokens.ipynb --output ../docs/intro-with-tokens-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/towards-embeddings.ipynb --output ../docs/towards-embeddings-render.ipynb
	mkdocs build --clean --site-dir public

serve-docs:
	jupyter nbconvert --to notebook --execute notebooks/intro-with-tokens.ipynb --output ../docs/intro-with-tokens-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/towards-embeddings.ipynb --output ../docs/towards-embeddings-render.ipynb
	mkdocs serve

gh-pages: docs
	git subtree push --prefix public origin gh-pages