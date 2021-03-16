v0.6.3

- Removed the deprecated `plot_correlation`
- Added the `plot_brush` tool.

v0.6.2

- Made each sklearn compatible language also apply `.partial_fit`

v0.6.1

- Added support for DIET embeddings
- Added support for spaCy 3.0
- Dropped support for BERT queries

v0.6.0

- Removed all transformers besides the ones in sklearn and umap

v0.5.10

- Added a helper to reverse strings for Arabic matplotlib charts.
- Put back ConveRT. It seems we only need a new hosting link.
- Added docs example for an Arabic benchmark.
- Added direct support for LaBSE.

v0.5.4

- Deprecated the `ConveRTLanguage` backend. The original authors removed the embeddings.
- Added the support for the Universal Sentence Encoder.

v0.5.3

- Fixed the `ConveRTLanguage` backend. The original source changed their download url.

v0.5.2

- Added tests for `matplotlib` and `altair`.
- Added `plot_3d`, allowing you to make some 3d visualisations.
- Added `assign` as a nicer alternative for `add_property`.
- Added a citation to an research paper on this library.
- Removed the "helper vectors" from our transformers.

v0.5.1

- Added a guide on debiasing on the docs.
- You can now specify the plot axes and title.
- We've added sensible default values to "plot_interactive".
We now assume that you want to plot the first two elements of a vector.

```python
# Before
emb.transform(Pca(2)).plot_interactive('pca_0', 'pca_1')
# After
emb.transform(Pca(2)).plot_interactive()
```

v0.5.0

- Added some robustness to the `matplotlib` based arrow and scatter charts.
- Started deprecating the `plot_correlation` method in favor
of the new `plot_distance` and `plot_similarity` methods.

v0.4.7

- Fixed bugs relating to conditional imports.
- Added a new `pipe` method.

v0.4.6

- Fixed bugs to become spaCy 2.3 compatible.
- Added scikit-learn pipeline compatibility for all models.

v0.4.5

- Created support for tfhub and huggingface backends.
- Added the `ivis` transformer.

v0.4.4

- Added support for ConveRT Embeddings.

v0.4.3

- Added support for similarity retreival for `CountVectorLang`
- Added more methods for `Embedding` objects: `distance`, `norm`

v0.4.2

- Added support the `gensim` language backend.

v0.4.1

- Added support for `TSNE`

v0.4.0

- Many small updates to improve documentation
- Fixed many small bugs for the `BytePairLanguage`






