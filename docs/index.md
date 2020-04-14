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

## Local Development

If you want to develop locally you can start by running this command after cloning.

```bash
make develop
```
