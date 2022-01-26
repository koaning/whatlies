# WhatLies

<img src="square-logo.svg" width=150 height=150 align="right">

A library that tries help you to understand. "What lies in word embeddings?"

## Produced

This project was initiated at [Rasa](https://rasa.com) as a fun side project
that supports the research and developer advocacy teams at Rasa.

## What it Does

This small library offers tools to make visualisation easier of both
word embeddings as well as operations on them. This should be considered
an experimental project.

This library will allow you to make visualisations of transformations
of word embeddings. Some of these transformations are linear algebra
operators.

<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script>

<small>Note that these charts are fully interactive. Click. Drag. Zoom in. Zoom out.</small>

<div id="vis1"></div>

But we also support other operations. Like **pca**  and **umap**;

<small>Just like before. Click. Drag. Zoom in. Zoom out.</small>

<div id="vis2"></div>

<script src="interactive1.js"></script>
<script src="interactive2.js"></script>


## Installation

You can install the package via pip;

```bash
pip install whatlies
```

This will install the base dependencies. Depending on the
transformers and language backends that you'll be using you
may want to install more. Here's all the possible installation
settings you could go for.

```bash
pip install whatlies[base]
pip install whatlies[umap]
pip install whatlies[spacy]
pip install whatlies[tfhub]
pip install whatlies[transformers]
pip install whatlies[sense2vec]
```

If you want it all you can also install via;

```bash
pip install whatlies[all]
```

Note that this will install dependencies but it
**will not** install all the language models you might
want to visualise. For example, you might still
need to manually download spaCy models if you intend
to use that backend.

## Similar Projects

There are some similar projects out and we figured it fair to mention and compare them here.

<details>
  <summary>Julia Bazińska & Piotr Migdal Web App</summary>
    <p>The original inspiration for this project came from <a href="https://lamyiowce.github.io/word2viz/">this web app</a>
    and <a href="https://www.youtube.com/watch?v=AGgCqpouKSs">this pydata talk</a>. It is a web app that takes a
    while to load but it is really fun to play with. The goal of this project is to make it easier to make similar
    charts from jupyter using different language backends.</p>
</details>

<details>
    <summary>Tensorflow Projector</summary>
    <p>From google there's the <a href="https://projector.tensorflow.org/">tensorflow projector project</a>. It offers
    highly interactive 3d visualisations as well as some transformations via tensorboard.</p>
    <ul>
    <li>The tensorflow projector will create projections in tensorboard, which you can also load
    into jupyter notebook but whatlies makes visualisations directly.</li>
    <li>The tensorflow projector supports interactive 3d visuals, which whatlies currently doesn't.</li>
    <li>Whatlies offers lego bricks that you can chain together to get a visualisation started. This
    also means that you're more flexible when it comes to transforming data before visualising it.</li>
    </ul>
</details>

<details>
    <summary>Parallax</summary>
    <p>From Uber AI Labs there's <a href="https://github.com/uber-research/parallax">parallax</a> which is described
    in a paper <a href="https://arxiv.org/abs/1905.12099">here</a>. There's a common mindset in the two tools;
    the goal is to use arbitrary user defined projections to understand embedding spaces better.
    That said, some differences that are worth to mention.</p>
    <ul>
    <li>It relies on bokeh as a visualisation backend and offers a lot of visualisation types
    (like radar plots). Whatlies uses altair and tries to stick to simple scatter charts.
    Altair can export interactive html/svg but it will not scale as well if you've drawing
    many points at the same time.</li>
    <li>Parallax is meant to be run as a stand-alone app from the command line while Whatlies is
    meant to be run from the jupyter notebook.</li>
    <li>Parallax gives a full user interface while Whatlies offers lego bricks that you can chain
    together to get a visualisation started.</li>
    <li>Whatlies relies on language backends (like spaCy, huggingface) to fetch word embeddings.
    Parallax allows you to instead fetch raw files on disk.</li>
    <li>Parallax has been around for a while, Whatlies is more new and therefore more experimental.</li>
    </ul>
</details>

### Citation

Please use the following citation when you found `whatlies` helpful for any of your work (find the `whatlies` paper [here](https://www.aclweb.org/anthology/2020.nlposs-1.8)):

```
@inproceedings{warmerdam-etal-2020-going,
    title = "Going Beyond {T}-{SNE}: Exposing whatlies in Text Embeddings",
    author = "Warmerdam, Vincent  and
      Kober, Thomas  and
      Tatman, Rachael",
    booktitle = "Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlposs-1.8",
    doi = "10.18653/v1/2020.nlposs-1.8",
    pages = "52--60",
    abstract = "We introduce whatlies, an open source toolkit for visually inspecting word and sentence embeddings. The project offers a unified and extensible API with current support for a range of popular embedding backends including spaCy, tfhub, huggingface transformers, gensim, fastText and BytePair embeddings. The package combines a domain specific language for vector arithmetic with visualisation tools that make exploring word embeddings more intuitive and concise. It offers support for many popular dimensionality reduction techniques as well as many interactive visualisations that can either be statically exported or shared via Jupyter notebooks. The project documentation is available from https://koaning.github.io/whatlies/.",
}
```


## Local Development

If you want to develop locally you can start by running this command after cloning.

```bash
make develop
```
