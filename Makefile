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

render-notebooks:
	nbstripout notebooks/*.ipynb
	jupyter nbconvert --to notebook --execute notebooks/01-intro-with-tokens.ipynb --output ../docs/intro-with-tokens-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/02-towards-embeddings.ipynb --output ../docs/towards-embeddings-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/03-interactive-transformations.ipynb --output ../docs/interactive-transformations-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/04-more-tokens-and-context.ipynb --output ../docs/more-tokens-and-context-render.ipynb
	jupyter nbconvert --to notebook --execute notebooks/05-language-backends.ipynb --output ../docs/language-backends-render.ipynb

test-notebooks:
	pytest --nbval --nbval-lax --disable-warnings notebooks/*.ipynb

docs: test-notebooks render-notebooks
	mkdocs build --clean --site-dir public

serve-docs: render-notebooks
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