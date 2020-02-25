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

render-notebooks:
	nbstripout notebooks/*.ipynb
	jupyter nbconvert --to notebook --execute notebooks/intro-with-tokens.ipynb --output ../docs/intro-with-tokens-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/towards-embeddings.ipynb --output ../docs/towards-embeddings-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/other-transformations.ipynb --output ../docs/other-transformations-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/other-visualisations.ipynb --output ../docs/other-visualisations-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/selecting-backends.ipynb --output ../docs/selecting-backends-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/embeddings-with-context.ipynb --output ../docs/embeddings-with-context-render.ipynb

test-notebooks:
	pytest --nbval --disable-warnings notebooks/*.ipynb

docs: test-notebooks render-notebooks
	mkdocs build --clean --site-dir public

serve-docs: render-notebooks
	mkdocs serve

gh-pages: docs
	git subtree push --prefix public origin gh-pages

clean:
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache