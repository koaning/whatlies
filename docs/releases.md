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

- Added some robustness tot he `matplotlib` based arrow and scatter charts.
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






