# whatlies 

<img src="logo.png">

A library that tries help you to understand. "What lies in word embeddings?"

## Produced 

<img src="square-logo.svg" width=150 height=150 align="right">

This project was initiated at [Rasa](https://rasa.com) as a fun side project
that supports the research and developer advocacy teams at Rasa. 

It is maintained by Vincent D. Warmerdam, Research Advocate at Rasa.

## What it Does

This small library offers tools to make visualisation easier of both
word embeddings as well as operations on them. This should be considered
an alpha project.

This library will allow you to make visualisations of transformations
of word embeddings. Some of these transformations are linear algebra
operators. 

<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script> 

<div id="vis1"></div>

But we also support other operations. Like **pca**  and **umap**;

<div id="vis2"></div>

<script src="interactive1.js"></script>
<script src="interactive2.js"></script>


## Installation 

For now we allow for installation with pip but only via git.

```bash
pip install git+git@github.com:RasaHQ/whatlies.git
```

## Local Development

If you want to develop locally you can start by running this command. 

```bash
make develop
```
