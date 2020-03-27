![](https://img.shields.io/pypi/v/whatlies)
![](https://img.shields.io/pypi/pyversions/whatlies)
![](https://img.shields.io/github/license/rasahq/whatlies)

<img src="docs/logo.png" width=255 height=255 align="right">

# whatlies 

A library that tries help you to understand (note the pun). 

> "What lies in word embeddings?"

This small library offers tools to make visualisation easier of both
word embeddings as well as operations on them. This should be considered
an experimental project that is in preview mode. 

Feedback is welcome. 

<img src="docs/square-logo.svg" width=200 height=200 align="right">

## Produced 

This project was initiated at [Rasa](https://rasa.com) as a fun side project
that supports the research and developer advocacy teams at Rasa. 
It is maintained by Vincent D. Warmerdam, Research Advocate at Rasa.

## Features

The idea is that you can load embeddings from a language backend 
and use mathematical operations on it. 

```python
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage

lang = SpacyLanguage("en_core_web_md")
words = ["cat", "dog", "fish", "kitten", "man", "woman", 
         "king", "queen", "doctor", "nurse"]

emb = EmbeddingSet(*[lang[w] for w in words])
emb.plot_interactive(x_axis=emb["man"], y_axis=emb["woman"])
```

![](docs/gif-zero.gif)

You can even do fancy operations. Like projecting unto and away
from vector embeddings! You can perform these on embeddings as 
well as sets of embeddings.  In the example below we attempt
to filter away gender bias using linear algebra operations.

```python
orig_chart = emb.plot_interactive('man', 'woman')

new_ts = emb | (emb['king'] - emb['queen'])
new_chart = new_ts.plot_interactive('man', 'woman')
```

![](docs/gif-one.gif)

There's also things like **pca** and **umap**.

```python
from whatlies.transformers import pca, umap

orig_chart = emb.plot_interactive('man', 'woman')
pca_plot = emb.transform(pca(2)).plot_interactive('pca_0', 'pca_1')
umap_plot = emb.transform(umap(2)).plot_interactive('umap_0', 'umap_1')

pca_plot | umap_plot
```

![](docs/gif-two.gif)

We even allow for BERT-style embeddings. Just use the square brackets. 

```python
lang = SpacyLanguage("en_trf_robertabase_lg")
lang['programming in [python]']
```

You'll now get the embedding for the token "python" but in context of "programming in python".

## Documentation 

To learn more and for a getting started guide, check out the [documentation](https://rasahq.github.io/whatlies/).

## Installation 

To install the package as well as all the dependencies, simply run;

```bash
pip install whatlies
```

## Local Development

If you want to develop locally you can start by running this command. 

```bash
make develop
```

### Documentation 

This is generated via

```
make docs
```
