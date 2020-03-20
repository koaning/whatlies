<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script> 

## Sets of Embeddings

The `Embedding` object merely has support for matplotlib, but the 
`EmbeddingSet` has support for interactive tools. It is also more
convenient. You can create an 

### Direct Creation 

You can create these objects directly. 

```python
import spacy 
from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet

nlp = spacy.load("en_core_web_md")

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman", 
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire", 
         "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water", 
         "person", "family", "brother", "sister"]

emb = EmbeddingSet({t.text: Embedding(t.text, t.vector) for t in nlp.pipe(words)})
```

This can be especially useful if you're creating your own embeddings. 

### Via Languages

But odds are that you just want to grab a language model from elsewhere. 
We've added backends to our library and this can be a convenient method
of getting sets of embeddings (typically more performant too).

```python
from whatlies.language import SpacyLanguage

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman", 
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire", 
         "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water", 
         "person", "family", "brother", "sister"]

lang = SpacyLanguage("en_core_web_md")
emb = lang[words]
```

## Plotting 

Either way, with an `EmbeddingSet` you can create meaningful interactive charts.

```python
emb.plot_interactive('man', 'woman')
```

<div id="vis1"></div>

<script>
fetch('tut2-chart1.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis1', out);
})
.catch(err => { throw err });
</script>

We can also retreive embeddings from the embeddingset. 

```python
emb['king']
```

Remember the operations we did before? We can also do that on these sets! 

```python
new_emb = emb | (emb['king'] - emb['queen'])
new_emb.plot_interactive('man', 'woman')
```

<div id="vis2"></div>

<script>
fetch('tut2-chart2.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis2', out);
})
.catch(err => { throw err });
</script>

### Combining Charts 

Often you'd like to compare the effect of a mapping. Since we make our interactive
charts with altair we get a nice api to stack charts next to eachother. 

```python
orig_chart = emb.plot_interactive('man', 'woman')
new_chart = new_emb.plot_interactive('man', 'woman')
orig_chart | new_chart
```

<div id="vis3"></div>

<script>
fetch('tut2-chart3.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis3', out);
})
.catch(err => { throw err });
</script>


You may have noticed that these charts appear in the documentation, fully interactively. 
This is another nice feature of Altair, the charts can be serialized in a json format and
hosted on the web.

## More Transformation

But there are more transformations that we might visualise. Let's demonstrate two here. 

```python
from whatlies.transformers import pca, umap

orig_chart = emb.plot_interactive('man', 'woman')
pca_emb = emb.transform(pca(2))
umap_emb = emb.transform(umap(2))
```

The transform method is able to take a transformation, let's say `pca(2)` and this will change
the embeddings in the set. It might also create new embeddings. In case of `pca(2)` it will 
also add two embeddings which represent the principal components. This is nice because
that means that we can plot along those axes.

```python
plot_pca = pca_emb.plot_interactive('pca_0', 'pca_1') 
plot_umap = umap_emb.plot_interactive('umap_0', 'umap_1')
plot_pca | plot_umap
```

<div id="vis4"></div>

<script>
fetch('tut2-chart4.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis4', out);
})
.catch(err => { throw err });
</script>

### More Components 

Suppose now that we'd like to visualise three principal components. We could do this.

```python
pca_emb = emb.transform(pca(3))
p1 = pca_emb.plot_interactive('pca_0', 'pca_1') 
p2 = pca_emb.plot_interactive('pca_2', 'pca_1')
p1 | p2
```

<div id="vis5"></div>

<script>
fetch('tut2-chart5.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis5', out);
})
.catch(err => { throw err });
</script>

### More Charts 

Let's not draw two components at a time, let's draw all of them.

```python
pca_emb.plot_interactive_matrix('pca_0', 'pca_1', 'pca_2')
```
<div id="vis6"></div>

<script>
fetch('tut2-chart6.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis6', out);
})
.catch(err => { throw err });
</script>

Zoom in on that chart. Don't forget to click and drag. Can we interpret the components?

