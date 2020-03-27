<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script>

## State and Colors

A goal of this package is to be able to compare the effect of transformations.

That is why some of our transformations carry state. Umap is one such example.

```python
from whatlies.language import SpacyLanguage
lang = SpacyLanguage('en_core_web_sm')

words1 = ["dog", "cat", "mouse", "deer", "elephant", "zebra", "fish",
          "rabbit", "rat", "tomato", "banana", "coffee", "tea", "apple", "union"]
words2 = ["run", "swim", "dance", "sit", "eat", "hear", "look", "run", "stand"]

umap = Umap(2)
emb1 = lang[words1].transform(umap).add_property('set', lambda d: 'set-one')
emb2 = lang[words2].transform(umap).add_property('set', lambda d: 'set-two')

both = emb1.merge(emb2)
```

In this code the transformer is trained on `emb1` and applied on both `emb1` and `emb2`.
We use the `.add_property` helper to indicate from which set the embeddings came.
This way we can use it as a color in an interactive plot.

```python
both.plot_interactive('umap_0', 'umap_1', color='set')
```

<div id="vis1"></div>

<script>
fetch('colors.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis1', out);
})
.catch(err => { throw err });
</script>
