# WhatLies

<img src="logo.png">

A library that tries help you to understand. "What lies in word embeddings?"

## Brief Introduction

If you prefer a video tutorial before reading the getting started guide watch this;

<iframe width="100%" height=450 src="https://www.youtube-nocookie.com/embed/FwkwC7IJWO0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Produced

<img src="square-logo.svg" width=150 height=150 align="right">

This project was initiated at [Rasa](https://rasa.com) as a fun side project
that supports the research and developer advocacy teams at Rasa.

It is maintained by Vincent D. Warmerdam, Research Advocate at Rasa.

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

This will install dependencies but it **won't** install all the language models you might want to visualise.

## Similar Projects

There are some projects out there who are working on similar tools and we figured it fair to mention and compare them here.

##### Tensorflow Projector

From google there's the [tensorflow projector project](https://projector.tensorflow.org/). It offers
highly interactive 3d visualisations as well as some transformations via tensorboard.

- The tensorflow projector will create projections in tensorboard, which you can also load
into jupyter notebook but whatlies makes visualisations directly.
- The tensorflow projector supports interactive 3d visuals, which whatlies currently doesn't.
- Whatlies offers lego bricks that you can chain together to get a visualisation started. This
also means that you're more flexible when it comes to transforming data before visualising it.

##### Parallax

From Uber AI Labs there's [parallax](https://github.com/uber-research/parallax) which is described
in a paper [here](https://arxiv.org/abs/1905.12099). There's a common mindset in the two tools;
the goal is to use arbitrary user defined projections to understand embedding spaces better.
That said, some differences that are worth to mention.

- It relies on bokeh as a visualisation backend and offers a lot of visualisation types
(like radar plots). Whatlies uses altair and tries to stick to simple scatter charts.
Altair can export interactive html/svg but it will not scale as well if you've drawing
many points at the same time.
- Parallax is meant to be run as a stand-alone app from the command line while Whatlies is
meant to be run from the jupyter notebook.
- Parallax gives a full user interface while Whatlies offers lego bricks that you can chain
together to get a visualisation started.
- Whatlies relies on language backends (like spaCy, huggingface) to fetch word embeddings.
Parallax allows you to instead fetch raw files on disk.
- Parallax has been around for a while, Whatlies is more new and therefore more experimental.

## Local Development

If you want to develop locally you can start by running this command after cloning.

```bash
make develop
```
