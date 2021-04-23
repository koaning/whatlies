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
         "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
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
         "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
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
from whatlies.transformers import Pca, Umap

orig_chart = emb.plot_interactive('man', 'woman')
pca_emb = emb.transform(Pca(2))
umap_emb = emb.transform(Umap(2))
```

The transform method is able to take a transformation, let's say `pca(2)` and this will change
the embeddings in the set. It might also create new embeddings. In case of `pca(2)` it will
also add two embeddings which represent the principal components. This is nice because
that means that we can plot along those axes.

```python
plot_pca = pca_emb.plot_interactive()
plot_umap = umap_emb.plot_interactive()
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

### Adding Color to the Charts

Sometimes it might be helpful to add color to the charts. In these situations we first need
to add a property to the embeddings in the embeddingset. This property can then be picked up
by a chart in order to make a subset stand out from the rest of the group.

```python
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
         "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
         "person", "family", "brother", "sister"]

colors = ["red", "blue",  "green", "yellow"]

lang = SpacyLanguage("en_core_web_md")

# Notice the `assign` method, this is where we assign the `is_color` property
# to each embedding in the embeddingset based on the "name".
embset = (lang[words]
            .transform(Pca(2))
            .assign(is_color=lambda e: e.name in colors))
embset.plot_interactive(color="is_color")
```

<div id="vis-color"></div>

<script>
fetch('tut-chart-color.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis-color', out);
})
.catch(err => { throw err });
</script>

### Using an Interactive Brush

We can also choose to use `plot_hover` instead of `plot_interactive`. The hover chart cannot
zoom in/out but it does allow you to draw a box to make a subselection. This can be very useful
when you're trying to get an overview of a cluster of embeddings.

```python
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
         "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
         "person", "family", "brother", "sister"]

colors = ["red", "blue",  "green", "yellow"]

lang = SpacyLanguage("en_core_web_md")
embset = (lang[words]
            .transform(Pca(2))
            .assign(is_color=lambda e: e.name in colors))
embset.plot_brush(n_show=15, color="is_color")
```

<div id="vis-hover"></div>

<script>
fetch('tut-chart-hover.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis-hover', out);
})
.catch(err => { throw err });
</script>

### Large Matrix Visualisations

If you're up for it, you can draw large matrices of charts too.

```python
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
         "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
         "person", "family", "brother", "sister"]

lang = SpacyLanguage("en_core_web_md")
lang[words].transform(Pca(2)).plot_interactive_matrix(0, 1, 2)
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
