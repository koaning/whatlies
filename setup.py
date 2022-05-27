import pathlib
from setuptools import setup, find_packages


base_packages = [
    "scikit-learn>=1.0.0",
    "altair>=4.2.0",
    "matplotlib>=3.5.0",
    "bpemb>=0.3.0",
    "gensim~=3.8.3",
]

umap_packages = [
    "umap-learn>=0.5.0",
]

fasttext_packages = [
    "fasttext~=0.9.1",
]

floret_packages = fasttext_packages + ["floret>=0.10.1"]

spacy_packages = [
    "spacy>=3.3.0",
    "spacy-lookups-data>=0.3.2",
]

s2v_packages = ["sense2vec>=2.0.0"] + spacy_packages

tf_packages = [
    "tensorflow<2.10,>=2.9.0",
    "tensorflow-text<2.10,>=2.9.0",
    "tensorflow-hub>=0.12.0,<1.0",
]

transformers_dep = [
    "transformers>=4.19.0",
]

sentence_tfm_dep = ["sentence-transformers>=2.2.0"]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
    "jupyterlab>=0.35.4",
    "nbstripout>=0.3.7",
    "nbval>=0.9.5",
]

test_packages = [
    "torch>=1.11.0",
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "nbval>=0.9.5",
    "pre-commit>=2.2.0",
]

all_deps = (
    tf_packages
    + transformers_dep
    + s2v_packages
    + sentence_tfm_dep
    + fasttext_packages
    + umap_packages
    + floret_packages
)
dev_packages = docs_packages + test_packages + all_deps


setup(
    name="whatlies",
    version="0.7.0",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Tools to help uncover `whatlies` in word embeddings.",
    long_description=pathlib.Path("readme.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://koaning.github.io/whatlies/",
    project_urls={
        "Documentation": "https://koaning.github.io/whatlies/",
        "Source Code": "https://github.com/koaning/whatlies/",
        "Issue Tracker": "https://github.com/koaning/whatlies/issues",
    },
    install_requires=base_packages,
    extras_require={
        "base": base_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "test": test_packages,
        "floret": floret_packages,
        "umap": base_packages + umap_packages,
        "tfhub": base_packages + tf_packages,
        "sense2vec": base_packages + s2v_packages,
        "spacy": base_packages + spacy_packages,
        "transformers": base_packages + transformers_dep,
        "sentence_tfm": base_packages + sentence_tfm_dep,
        "all": all_deps,
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
