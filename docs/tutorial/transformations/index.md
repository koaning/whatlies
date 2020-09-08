<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script>

## State and Colors

A goal of this package is to be able to compare the effect of transformations.

That is why some of our transformations carry state. Umap is one such example.

```python
from whatlies.language.language import SpacyLanguage
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
both.plot_interactive(color='set')
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

## Visualising Differences

Let's create two embeddings.

```python
from whatlies.language.language import SpacyLanguage

lang = SpacyLanguage("en_core_web_md")
words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
         "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
         "person", "family", "brother", "sister", "happy prince", "sad prince"]

emb1 = lang[words]
emb2 = lang[words] | (lang["king"] - lang["queen"])
```

The two embeddings should be similar but we can show that they are different.

```python
p1 = emb1.plot_interactive("man", "woman")
p2 = emb2.plot_interactive("man", "woman")

p1 | p2
```

<div id="vis2"></div>

<script>
fetch('two-groups-one.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis2', out);
})
.catch(err => { throw err });
</script>

In this case, both plots will plot their embeddings with regards
to their own embedding for `man` and `woman`. But we can also
explicitly tell them to compare against the original vectors
from `emb1`.

```python
p1 = emb1.plot_interactive(emb1["man"], emb1["woman"])
p2 = emb2.plot_interactive(emb1["man"], emb1["woman"])

p1 | p2
```

<div id="vis3"></div>

<script>
fetch('two-groups-two.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis3', out);
})
.catch(err => { throw err });
</script>

It's subtle but it is important to recognize.

### Movement

If you want to highlight the movement that occurs because of
a transformation then you might prefer to show a movement plot.

```python
emb1.plot_movement(emb2, "man", "woman").properties(width=600, height=450)
```

<div id="vis4"></div>

<script>
fetch('movement.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis4', out);
})
.catch(err => { throw err });
</script>
